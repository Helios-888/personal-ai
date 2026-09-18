"""run_eval CLI: 依存（LLM 呼び出し・トークン計測・Open WebUI）を差し替えて流れを検証する。"""
import hashlib
import json
import signal
from pathlib import Path

import pytest
import requests
import yaml

from scripts.lib.freeze import file_sha256, write_record
from scripts.lib.git_state import GitState
from scripts.lib.llm_client import ChatResult
from scripts.lib.openwebui_client import OpenWebUIAuthError
from scripts.lib.sync import build_model_form
from scripts.lib.tokens import TokenCount
from scripts.run_eval import B0_SYSTEM_PROMPT, Target, run

AGENT_YAML = {
    "id": "kukai-ai",
    "name": "空海",
    "base_model_id": "kukai",
    "description": "d",
    "system_prompt": {"parts": ["agents/kukai/a.md", "agents/kukai/b.md"], "budget_tokens": 5000},
    "params": {"temperature": 0.3},
    "knowledge": [],
    "filters": [],
}
SYSTEM_PROMPT = "alpha\n\nbeta\n"  # a.md と b.md を build_system_prompt でつないだもの
QUESTIONS_YAML = """\
frozen: 2026-09-18
questions:
  - id: F01
    kind: factual
    text: 問一
    expect: answer
    key_points:
      - point: 正解一
        source: 出典一
  - id: H01
    kind: holdout
    episode: ep-kukai-0001
    text: |
      次の場面で。

      状況。
    expect: direction
    key_points:
      - point: 正解二
        source: 出典二
"""
H01_TEXT = "次の場面で。\n\n状況。"
TODAY = "2026-09-19"
ENV = {"OPENWEBUI_API_KEY": "secret-key"}
SAME_AS_REPO = object()


def make_repo(tmp_path: Path, agent: dict = AGENT_YAML) -> Path:
    agent_dir = tmp_path / "agents" / "kukai"
    agent_dir.mkdir(parents=True)
    (agent_dir / "a.md").write_text("alpha\n", encoding="utf-8")
    (agent_dir / "b.md").write_text("beta\n", encoding="utf-8")
    (agent_dir / "agent.yaml").write_text(yaml.safe_dump(agent, allow_unicode=True), encoding="utf-8")
    eval_dir = tmp_path / "evaluations" / "kukai"
    eval_dir.mkdir(parents=True)
    questions = eval_dir / "questions.yaml"
    questions.write_bytes(QUESTIONS_YAML.encode("utf-8"))
    write_record(questions)
    return tmp_path


def out_dir(root: Path, name: str = f"{TODAY}-B1") -> Path:
    return root / "evaluations" / "kukai" / "baseline" / name


CLEAN = GitState(commit="c0ffee", dirty=())


class RecordingChat:
    def __init__(self, *, finish="stop", reasoning="", prompt_tokens=None):
        self.calls = []
        self.finish = finish
        self.reasoning = reasoning
        self.prompt_tokens = prompt_tokens or (lambda call_number: 40)

    def __call__(self, base_url, model, *, system, user, temperature, max_tokens, endpoint, headers):
        self.calls.append(
            dict(
                base_url=base_url,
                model=model,
                system=system,
                user=user,
                temperature=temperature,
                max_tokens=max_tokens,
                endpoint=endpoint,
                headers=headers,
            )
        )
        return ChatResult(
            content=f"答 {user}",
            reasoning=self.reasoning,
            finish_reason=self.finish,
            completion_tokens=5,
            elapsed_seconds=1.5,
            prompt_tokens=self.prompt_tokens(len(self.calls)),
        )


class FailingChat(RecordingChat):
    """指定回数目の呼び出しで失敗を起こす（既定は通信失敗）。"""

    def __init__(self, fail_at: int, error: BaseException = requests.ConnectionError("open-webui down")):
        super().__init__()
        self.fail_at = fail_at
        self.error = error

    def __call__(self, *args, **kwargs):
        if len(self.calls) + 1 == self.fail_at:
            self.calls.append(kwargs)
            raise self.error
        return super().__call__(*args, **kwargs)


class FakeOpenWebUI:
    """登録済みのモデルを返すだけの Open WebUI。既定ではリポジトリの定義どおりに登録されている。"""

    def __init__(self, registered=SAME_AS_REPO):
        self.registered = build_model_form(AGENT_YAML, SYSTEM_PROMPT) if registered is SAME_AS_REPO else registered
        self.requested = []

    def get_model(self, model_id):
        self.requested.append(model_id)
        return self.registered


def run_cli(argv, root, *, chat=None, env=ENV, openwebui=None, tokens=TokenCount(30, True), git=CLEAN, git_calls=None):
    chat = chat if chat is not None else RecordingChat()
    openwebui = openwebui if openwebui is not None else FakeOpenWebUI()
    factory_calls, token_calls = [], []
    git_calls = git_calls if git_calls is not None else []

    def client_factory(url, api_key):
        factory_calls.append((url, api_key))
        return openwebui

    def token_counter(text, **kwargs):
        token_calls.append(dict(text=text, **kwargs))
        return tokens

    def git_state(repo_root, paths):
        git_calls.append((repo_root, list(paths)))
        return git

    rc = run(
        argv,
        root=root,
        env=env,
        chat=chat,
        token_counter=token_counter,
        client_factory=client_factory,
        git_state=git_state,
        today=TODAY,
    )
    return rc, chat, factory_calls, token_calls


def read_records(root: Path, name: str = f"{TODAY}-B1") -> list[dict]:
    lines = (out_dir(root, name) / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def read_answers(root: Path, name: str = f"{TODAY}-B1") -> tuple[dict, list[dict]]:
    records = read_records(root, name)
    return records[0], [r for r in records if r["record"] == "answer"]


# --- 条件と経路 ---


def test_b1_via_openwebui_asks_the_registered_model_without_sending_a_system_prompt(tmp_path):
    # Open WebUI は登録済みの system を要求の先頭に足す。こちらからも送ると二重になる
    root = make_repo(tmp_path)

    rc, chat, factory_calls, _ = run_cli(["--condition", "B1"], root)

    assert rc == 0
    assert factory_calls == [("http://localhost:3000", "secret-key")]
    assert chat.calls
    for call in chat.calls:
        assert call["base_url"] == "http://localhost:3000"
        assert call["endpoint"] == "/api/chat/completions"
        assert call["model"] == "kukai-ai"
        assert call["system"] is None
        assert call["headers"] == {"Authorization": "Bearer secret-key"}
        assert call["temperature"] == 0.3
        assert call["max_tokens"] == 800


def test_b0_sends_only_the_one_sentence_system_prompt_to_the_base_model(tmp_path):
    root = make_repo(tmp_path)

    rc, chat, factory_calls, _ = run_cli(["--condition", "B0"], root)

    assert rc == 0
    assert B0_SYSTEM_PROMPT == "あなたは空海（774–835）です。空海として一人称で答えてください。"  # 設計書の文言
    assert factory_calls == []  # 登録の照合は B1 だけ
    assert {c["model"] for c in chat.calls} == {"kukai"}
    assert {c["system"] for c in chat.calls} == {B0_SYSTEM_PROMPT}
    assert {c["endpoint"] for c in chat.calls} == {"/api/chat/completions"}
    assert {c["temperature"] for c in chat.calls} == {0.3}  # 両条件で同じ temperature


@pytest.mark.parametrize("condition, system", [("B0", B0_SYSTEM_PROMPT), ("B1", SYSTEM_PROMPT)])
def test_llama_swap_route_sends_the_system_prompt_itself_to_the_base_model(tmp_path, condition, system):
    root = make_repo(tmp_path)

    rc, chat, factory_calls, _ = run_cli(
        ["--condition", condition, "--route", "llama-swap"], root, env={"LLAMA_SWAP_URL": "http://llm:8080"}
    )

    assert rc == 0
    assert factory_calls == []  # API キーも Open WebUI も要らない
    call = chat.calls[0]
    assert (call["base_url"], call["endpoint"], call["model"]) == ("http://llm:8080", "/v1/chat/completions", "kukai")
    assert call["system"] == system
    assert call["headers"] is None


def test_openwebui_url_comes_from_env(tmp_path):
    root = make_repo(tmp_path)

    _, chat, _, _ = run_cli(["--condition", "B0"], root, env={**ENV, "OPENWEBUI_URL": "http://owui:3000"})

    assert chat.calls[0]["base_url"] == "http://owui:3000"


def test_each_question_is_asked_three_times_round_by_round(tmp_path):
    root = make_repo(tmp_path)

    _, chat, _, _ = run_cli(["--condition", "B1"], root)

    assert [c["user"] for c in chat.calls] == ["問一", H01_TEXT] * 3


def test_the_answer_key_is_never_sent_to_the_model(tmp_path):
    root = make_repo(tmp_path)

    _, chat, _, _ = run_cli(["--condition", "B1"], root)

    sent = json.dumps(chat.calls, ensure_ascii=False)
    assert "正解" not in sent
    assert "出典" not in sent


def test_system_prompt_tokens_are_counted_on_the_base_model(tmp_path, capsys):
    root = make_repo(tmp_path)

    _, _, _, token_calls = run_cli(["--condition", "B1"], root, env={**ENV, "LLAMA_SWAP_URL": "http://llm:8080"})

    assert token_calls == [dict(text=SYSTEM_PROMPT, base_url="http://llm:8080", model="kukai")]
    assert "30 tokens (exact)" in capsys.readouterr().out


def test_run_warns_when_the_token_count_is_only_an_estimate(tmp_path, capsys):
    root = make_repo(tmp_path)

    run_cli(["--condition", "B0"], root, tokens=TokenCount(30, False))

    assert "estimate" in capsys.readouterr().err


# --- 記録 ---


def test_answers_are_recorded_as_jsonl_under_a_header_that_pins_the_comparison_ground(tmp_path):
    root = make_repo(tmp_path)

    run_cli(["--condition", "B1"], root)

    header, answers = read_answers(root)
    assert header["record"] == "header"
    assert header["condition"] == "B1"
    assert header["route"] == "openwebui"
    assert header["url"] == "http://localhost:3000/api/chat/completions"
    assert header["model"] == "kukai-ai"
    assert header["questions_sha256"] == file_sha256(root / "evaluations" / "kukai" / "questions.yaml")
    assert header["system_prompt_sha256"] == hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert header["system_prompt_sent"] is False
    assert header["system_tokens"] == 30
    assert header["temperature"] == 0.3
    assert header["max_tokens"] == 800
    assert header["repeats"] == 3
    assert header["question_count"] == 2
    assert [(a["question_id"], a["repeat"]) for a in answers] == [
        ("F01", 1),
        ("H01", 1),
        ("F01", 2),
        ("H01", 2),
        ("F01", 3),
        ("H01", 3),
    ]
    assert {a["condition"] for a in answers} == {"B1"}
    assert answers[0]["content"] == "答 問一"


def test_b0_header_pins_the_one_sentence_prompt(tmp_path):
    root = make_repo(tmp_path)

    run_cli(["--condition", "B0"], root)

    header, _ = read_answers(root, f"{TODAY}-B0")
    assert header["system_prompt_sha256"] == hashlib.sha256(B0_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert header["system_prompt_sent"] is True
    assert header["model"] == "kukai"


def test_transcript_is_written_next_to_the_answers(tmp_path):
    root = make_repo(tmp_path)

    run_cli(["--condition", "B1"], root)

    text = (out_dir(root) / "transcript.md").read_text(encoding="utf-8")
    assert "## F01 [factual]" in text
    assert "### 3 回目" in text
    assert "答 問一" in text


def test_the_api_key_is_never_written_to_the_records(tmp_path):
    root = make_repo(tmp_path)

    run_cli(["--condition", "B1"], root)

    for path in out_dir(root).iterdir():
        assert "secret-key" not in path.read_text(encoding="utf-8"), path.name


def test_label_is_appended_to_the_folder_name(tmp_path):
    root = make_repo(tmp_path)

    run_cli(["--condition", "B0", "--label", "retry"], root)

    assert (out_dir(root, f"{TODAY}-B0-retry") / "answers.jsonl").is_file()


def test_out_dir_and_only_and_repeats_narrow_the_run_for_the_route_preflight(tmp_path):
    root = make_repo(tmp_path)

    rc, chat, _, _ = run_cli(
        ["--condition", "B1", "--only", "H01", "--repeats", "1", "--out-dir", "evaluations/kukai/preflight"], root
    )

    assert rc == 0
    assert [c["user"] for c in chat.calls] == [H01_TEXT]
    answers = root / "evaluations" / "kukai" / "preflight" / f"{TODAY}-B1" / "answers.jsonl"
    header = json.loads(answers.read_text(encoding="utf-8").splitlines()[0])
    assert (header["question_count"], header["repeats"]) == (1, 1)


def test_answers_collected_so_far_survive_a_failure(tmp_path, capsys):
    root = make_repo(tmp_path)

    with pytest.raises(requests.ConnectionError):
        run_cli(["--condition", "B1"], root, chat=FailingChat(fail_at=3))

    _, answers = read_answers(root)
    assert [(a["question_id"], a["repeat"]) for a in answers] == [("F01", 1), ("H01", 1)]
    assert read_records(root)[-1] == {
        "record": "end",
        "status": "interrupted",
        "answers": 2,
        "inconsistent_prompt_tokens": [],
    }
    text = (out_dir(root) / "transcript.md").read_text(encoding="utf-8")
    assert "途中で中断（2/6 回答まで）" in text
    assert "saved (partial)" in capsys.readouterr().err


def test_a_failure_on_the_first_call_still_leaves_a_marked_record(tmp_path):
    # API キーの誤りなどで 1 問目から失敗しても、「中断・0 件」と機械が読める形で残す
    root = make_repo(tmp_path)

    with pytest.raises(requests.ConnectionError):
        run_cli(["--condition", "B1"], root, chat=FailingChat(fail_at=1))

    records = read_records(root)
    assert [r["record"] for r in records] == ["header", "end"]
    assert (records[-1]["status"], records[-1]["answers"]) == ("interrupted", 0)
    assert "途中で中断（0/6 回答まで）" in (out_dir(root) / "transcript.md").read_text(encoding="utf-8")


def test_an_interrupt_is_recorded_as_interrupted(tmp_path):
    root = make_repo(tmp_path)

    with pytest.raises(KeyboardInterrupt):
        run_cli(["--condition", "B1"], root, chat=FailingChat(fail_at=4, error=KeyboardInterrupt()))

    assert read_records(root)[-1]["status"] == "interrupted"


def test_a_completed_run_ends_with_a_complete_marker(tmp_path):
    root = make_repo(tmp_path)

    run_cli(["--condition", "B1"], root)

    assert read_records(root)[-1] == {
        "record": "end",
        "status": "complete",
        "answers": 6,
        "inconsistent_prompt_tokens": [],
    }


def test_run_warns_when_the_input_size_of_a_question_changes_between_rounds(tmp_path, capsys):
    # 実行中に sync_openwebui.py を走らせるなどして、途中から別のシステムプロンプトで答えた回を見つける
    root = make_repo(tmp_path)
    grows_after_round_one = RecordingChat(prompt_tokens=lambda call_number: 40 if call_number <= 2 else 90)

    run_cli(["--condition", "B1"], root, chat=grows_after_round_one)

    assert "F01, H01" in capsys.readouterr().err
    assert read_records(root)[-1]["inconsistent_prompt_tokens"] == ["F01", "H01"]


def test_run_reports_the_truncation_count_and_warns_above_5_percent(tmp_path, capsys):
    root = make_repo(tmp_path)

    rc, _, _, _ = run_cli(["--condition", "B1"], root, chat=RecordingChat(finish="length"))

    captured = capsys.readouterr()
    assert rc == 0
    assert "finish=length: 6/6" in captured.out
    assert "5%" in captured.err
    assert "両条件" in captured.err  # 片方の条件だけ max_tokens を変えて取り直すと土俵が崩れる


def test_run_does_not_warn_when_no_answer_hits_the_output_limit(tmp_path, capsys):
    root = make_repo(tmp_path)

    run_cli(["--condition", "B1"], root)

    captured = capsys.readouterr()
    assert "finish=length: 0/6" in captured.out
    assert "warning" not in captured.err


def test_run_warns_when_reasoning_comes_back_although_thinking_is_off(tmp_path, capsys):
    root = make_repo(tmp_path)

    run_cli(["--condition", "B1"], root, chat=RecordingChat(reasoning="考え中"))

    assert "思考" in capsys.readouterr().err


# --- 送る前に止めるもの（終了コード 2、LLM は呼ばない） ---


def test_run_refuses_to_overwrite_an_existing_folder(tmp_path, capsys):
    root = make_repo(tmp_path)
    out_dir(root).mkdir(parents=True)

    rc, chat, _, _ = run_cli(["--condition", "B1"], root)

    assert rc == 2
    assert chat.calls == []
    assert f"{TODAY}-B1" in capsys.readouterr().err


def test_run_refuses_questions_changed_after_freezing(tmp_path, capsys):
    root = make_repo(tmp_path)
    questions = root / "evaluations" / "kukai" / "questions.yaml"
    questions.write_bytes(questions.read_bytes().replace("問一".encode("utf-8"), "問壱".encode("utf-8")))

    rc, chat, _, _ = run_cli(["--condition", "B1"], root)

    assert rc == 2
    assert chat.calls == []
    assert "凍結後に変更" in capsys.readouterr().err
    assert not out_dir(root).exists()


def test_b1_refuses_when_the_openwebui_model_differs_from_the_repo(tmp_path, capsys):
    root = make_repo(tmp_path)
    stale = build_model_form(AGENT_YAML, "old prompt")

    rc, chat, _, _ = run_cli(["--condition", "B1"], root, openwebui=FakeOpenWebUI(stale))

    assert rc == 2
    assert chat.calls == []
    err = capsys.readouterr().err
    assert "params" in err
    assert "sync_openwebui.py" in err


def test_b1_refuses_when_the_model_is_not_registered(tmp_path, capsys):
    root = make_repo(tmp_path)

    rc, chat, _, _ = run_cli(["--condition", "B1"], root, openwebui=FakeOpenWebUI(None))

    assert rc == 2
    assert chat.calls == []
    assert "登録されていません" in capsys.readouterr().err


def test_b1_refuses_an_agent_that_already_has_knowledge(tmp_path, capsys):
    root = make_repo(tmp_path, agent={**AGENT_YAML, "knowledge": ["kukai-primary"]})

    rc, chat, _, _ = run_cli(["--condition", "B1", "--route", "llama-swap"], root)

    assert rc == 2
    assert chat.calls == []
    assert "Knowledge" in capsys.readouterr().err


def test_openwebui_route_requires_an_api_key(tmp_path, capsys):
    root = make_repo(tmp_path)

    rc, chat, _, _ = run_cli(["--condition", "B0"], root, env={})

    assert rc == 2
    assert chat.calls == []
    assert "OPENWEBUI_API_KEY" in capsys.readouterr().err


def test_only_rejects_an_unknown_question_id(tmp_path, capsys):
    root = make_repo(tmp_path)

    rc, chat, _, _ = run_cli(["--condition", "B1", "--only", "F01,F99", "--out-dir", "evaluations/kukai/preflight"], root)

    assert rc == 2
    assert chat.calls == []
    assert "F99" in capsys.readouterr().err


@pytest.mark.parametrize("only", ["", ","])
def test_only_rejects_an_empty_list_instead_of_asking_everything(tmp_path, capsys, only):
    # --only "$IDS" で変数が空だったとき、先行確認のつもりで全 96 回を走らせない
    root = make_repo(tmp_path)

    rc, chat, _, _ = run_cli(["--condition", "B1", "--only", only, "--out-dir", "evaluations/kukai/preflight"], root)

    assert rc == 2
    assert chat.calls == []
    assert "--only" in capsys.readouterr().err


@pytest.mark.parametrize("narrowing", [["--only", "F01"], ["--repeats", "1"]], ids=["only", "repeats"])
def test_a_narrowed_run_is_not_saved_as_the_baseline(tmp_path, capsys, narrowing):
    root = make_repo(tmp_path)

    rc, chat, _, _ = run_cli(["--condition", "B1", *narrowing], root)

    assert rc == 2
    assert chat.calls == []
    assert "--out-dir" in capsys.readouterr().err


def test_the_baseline_requires_committed_definitions_and_code(tmp_path, capsys):
    root = make_repo(tmp_path)
    dirty = GitState(commit="c0ffee", dirty=(" M agents/kukai/b.md",))

    rc, chat, _, _ = run_cli(["--condition", "B1"], root, git=dirty)

    assert rc == 2
    assert chat.calls == []
    err = capsys.readouterr().err
    assert "未コミット" in err
    assert "agents/kukai/b.md" in err


def test_a_run_outside_the_baseline_records_uncommitted_changes_instead_of_refusing(tmp_path):
    root = make_repo(tmp_path)
    dirty = GitState(commit="c0ffee", dirty=(" M agents/kukai/b.md",))

    rc, _, _, _ = run_cli(["--condition", "B1", "--out-dir", "evaluations/kukai/preflight"], root, git=dirty)

    assert rc == 0
    answers = root / "evaluations" / "kukai" / "preflight" / f"{TODAY}-B1" / "answers.jsonl"
    header = json.loads(answers.read_text(encoding="utf-8").splitlines()[0])
    assert (header["git_commit"], header["git_dirty"]) == ("c0ffee", [" M agents/kukai/b.md"])


@pytest.mark.parametrize(
    "condition, expected",
    [
        ("B0", ["agents/kukai/agent.yaml", "evaluations/kukai/questions.yaml", "scripts"]),
        (
            "B1",
            [
                "agents/kukai/agent.yaml",
                "evaluations/kukai/questions.yaml",
                "scripts",
                "agents/kukai/a.md",
                "agents/kukai/b.md",
            ],
        ),
    ],
)
def test_git_state_is_checked_for_what_the_condition_depends_on(tmp_path, condition, expected):
    root = make_repo(tmp_path)
    git_calls = []

    run_cli(["--condition", condition], root, git_calls=git_calls)

    assert git_calls == [(root, expected)]
    header, _ = read_answers(root, f"{TODAY}-{condition}")
    assert header["git_commit"] == "c0ffee"


def test_openwebui_route_refuses_a_system_prompt_with_template_variables(tmp_path, capsys):
    # Open WebUI は {{CURRENT_DATE}} などを置き換えるので、記録の指紋と実際に届くものがずれる
    root = make_repo(tmp_path)
    (root / "agents" / "kukai" / "a.md").write_text("今日は {{CURRENT_DATE}}\n", encoding="utf-8")
    registered = build_model_form(AGENT_YAML, "今日は {{CURRENT_DATE}}\n\nbeta\n")  # 登録の照合は通る状態にする

    rc, chat, _, _ = run_cli(["--condition", "B1"], root, openwebui=FakeOpenWebUI(registered))

    assert rc == 2
    assert chat.calls == []
    assert "{{" in capsys.readouterr().err


def test_llama_swap_route_sends_template_variables_as_they_are(tmp_path):
    root = make_repo(tmp_path)
    (root / "agents" / "kukai" / "a.md").write_text("今日は {{CURRENT_DATE}}\n", encoding="utf-8")

    rc, _, _, _ = run_cli(["--condition", "B1", "--route", "llama-swap"], root)

    assert rc == 0


def test_a_network_error_while_checking_the_registration_is_not_an_input_error(tmp_path):
    # 入力の不備（終了コード 2）と、つながらない（main で 1）を取り違えない
    root = make_repo(tmp_path)

    class Unreachable(FakeOpenWebUI):
        def get_model(self, model_id):
            raise requests.ConnectionError("refused")

    with pytest.raises(requests.ConnectionError):
        run_cli(["--condition", "B1"], root, openwebui=Unreachable())


def test_max_tokens_is_sent_and_recorded_as_given(tmp_path):
    root = make_repo(tmp_path)

    _, chat, _, _ = run_cli(["--condition", "B1", "--max-tokens", "640"], root)

    assert {c["max_tokens"] for c in chat.calls} == {640}
    header, _ = read_answers(root)
    assert header["max_tokens"] == 640


def test_the_api_key_does_not_appear_in_the_target_repr():
    target = Target("http://x", "/api/chat/completions", "m", {"Authorization": "Bearer secret-key"}, None)

    assert "secret-key" not in repr(target)


def test_run_rejects_a_label_that_could_escape_the_output_folder(tmp_path, capsys):
    root = make_repo(tmp_path)

    rc, chat, _, _ = run_cli(["--condition", "B1", "--label", "../x"], root)

    assert rc == 2
    assert chat.calls == []


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--condition", "B2"],
        ["--condition", "B1", "--route", "direct"],
        ["--condition", "B1", "--repeats", "0"],
        ["--condition", "B1", "--max-tokens", "0"],
    ],
    ids=["no-condition", "unknown-condition", "unknown-route", "zero-repeats", "zero-max-tokens"],
)
def test_run_rejects_invalid_arguments(tmp_path, argv):
    root = make_repo(tmp_path)

    with pytest.raises(SystemExit) as excinfo:
        run_cli(argv, root)

    assert excinfo.value.code == 2


# --- main ---


@pytest.fixture(autouse=True)
def installed_signal_handlers(monkeypatch):
    """main() が pytest 自身のシグナル処理を書き換えないよう、登録を記録するだけにする。"""
    installed = {}
    monkeypatch.setattr(signal, "signal", lambda signum, handler: installed.__setitem__(signum, handler))
    return installed


def test_main_merges_dotenv_with_environ_and_wires_real_dependencies(tmp_path, monkeypatch):
    import scripts.run_eval as cli
    from scripts.lib.git_state import git_state
    from scripts.lib.llm_client import chat_completion
    from scripts.lib.openwebui_client import OpenWebUIClient
    from scripts.lib.tokens import count_tokens

    root = make_repo(tmp_path)
    (root / ".env").write_text("OPENWEBUI_API_KEY=fromfile\nOPENWEBUI_URL=http://file:1\n", encoding="utf-8")
    monkeypatch.setenv("OPENWEBUI_URL", "http://environ:2")
    seen = {}

    def fake_run(argv, *, root, env, chat, token_counter, client_factory, git_state, today):
        seen.update(argv=argv, env=env, chat=chat, token_counter=token_counter, client_factory=client_factory)
        seen.update(git_state=git_state, today=today)
        return 0

    monkeypatch.setattr(cli, "run", fake_run)

    assert cli.main(["--condition", "B1"], root=root) == 0
    assert seen["argv"] == ["--condition", "B1"]
    assert seen["env"]["OPENWEBUI_API_KEY"] == "fromfile"
    assert seen["env"]["OPENWEBUI_URL"] == "http://environ:2"
    assert seen["chat"] is chat_completion
    assert seen["token_counter"] is count_tokens
    assert seen["client_factory"] is OpenWebUIClient
    assert seen["git_state"] is git_state
    assert len(seen["today"]) == len("2026-09-19")


def test_main_turns_hangup_and_terminate_into_an_interrupt(tmp_path, monkeypatch, installed_signal_handlers):
    # ssh が切れたとき（SIGHUP）や kill（SIGTERM）でも、中断の記録を書いてから止まるように
    import scripts.run_eval as cli

    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: 0)

    cli.main(["--condition", "B1"], root=make_repo(tmp_path))

    assert installed_signal_handlers[signal.SIGTERM] is cli.raise_interrupt
    assert installed_signal_handlers[signal.SIGHUP] is cli.raise_interrupt
    with pytest.raises(KeyboardInterrupt):
        cli.raise_interrupt(signal.SIGTERM, None)


def test_main_reports_an_interrupt_in_one_line_with_exit_130(tmp_path, monkeypatch, capsys):
    import scripts.run_eval as cli

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "run", interrupted)

    assert cli.main(["--condition", "B1"], root=make_repo(tmp_path)) == 130
    err = capsys.readouterr().err
    assert "interrupted" in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    "error",
    [requests.ConnectionError("refused"), OpenWebUIAuthError("refused")],
    ids=["network", "openwebui-auth"],
)
def test_main_turns_network_and_openwebui_errors_into_one_line_and_exit_1(tmp_path, monkeypatch, capsys, error):
    import scripts.run_eval as cli

    root = make_repo(tmp_path)

    def explode(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "run", explode)

    assert cli.main(["--condition", "B1"], root=root) == 1
    err = capsys.readouterr().err
    assert "refused" in err
    assert "Traceback" not in err
