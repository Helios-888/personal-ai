"""fetch_sat CLI: SAT の詳細ページを続きのリンクでたどり、範囲を絞り、伏せる箇所を削って Knowledge 用の Markdown を書く流れ。

ネットワークには出ない。ページは偽の get が返し、待ちは記録するだけ。
"""
from pathlib import Path

import pytest

import scripts.fetch_sat as fetch_sat
from scripts.fetch_sat import BASE, fetch_work, run

EXCLUSIONS_YAML = """\
terms: [南山]
cut:
  - file: later-attribution/later-attribution__御遺告.md
    first: T2431_.77.0409b21
    last: T2431_.77.0409b22
    reason: 伏せた場面の判断
kept: []
"""


def line(line_id: str, body: str) -> str:
    return f'<span style="color:black">{line_id}: </span><a name="{line_id[-7:]}">{body}</a><br/>'


def page(*lines: tuple[str, str], following: str = "") -> str:
    tail = (f'<div id="nextl" onclick="location.href=\'ddb-sat2.php?mode=detail&nonum=&kaeri=&mode2=2&useid={following}\'">'
            "</div>") if following else ""
    return "".join(line(i, b) for i, b in lines) + tail


class FakeSite:
    """URL → ページ。問われた URL と待ちの秒数を記録する。"""

    def __init__(self, pages: dict[str, str]):
        self.pages = pages
        self.asked: list[str] = []
        self.waits: list[float] = []

    def get(self, url: str) -> str:
        self.asked.append(url)
        return self.pages[url]

    def wait(self, seconds: float) -> None:
        self.waits.append(seconds)


def first_url(number: str) -> str:
    return f"{BASE}?mode=detail&useid={number[1:]}_"


def next_url(useid: str) -> str:
    return f"{BASE}?mode=detail&nonum=&kaeri=&mode2=2&useid={useid}"


A1, A2, A3 = "T2428_.77.0381b16", "T2428_.77.0381b17", "T2428_.77.0381b18"


def test_fetch_work_follows_the_continuation_until_the_last_page():
    site = FakeSite({
        first_url("T2428"): page((A1, "即身成佛義"), following="2428_,77,0381b17"),
        next_url("2428_,77,0381b17"): page((A1, "即身成佛義"), (A2, "問曰。"), following="2428_,77,0381b18"),
        next_url("2428_,77,0381b18"): page((A2, "問曰。"), (A3, "諸經論中")),
    })
    url, lines = fetch_work("T2428", site.get, site.wait)
    assert url == first_url("T2428")
    assert lines == [(A1, "即身成佛義"), (A2, "問曰。"), (A3, "諸經論中")]  # 伸びた表示の重なりは 1 回だけ
    assert site.waits == [fetch_sat.WAIT_SECONDS] * 2  # ページの間に 1 秒ずつ


def test_fetch_work_stops_when_the_continuation_goes_round():
    site = FakeSite({
        first_url("T2428"): page((A1, "一"), following="2428_,77,0381b17"),
        next_url("2428_,77,0381b17"): page((A2, "二"), following="2428_,77,0381b17"),
    })
    with pytest.raises(ValueError, match="回っています"):
        fetch_work("T2428", site.get, site.wait)


def test_fetch_work_gives_up_after_the_page_limit(monkeypatch):
    monkeypatch.setattr(fetch_sat, "MAX_PAGES", 2)
    site = FakeSite({
        first_url("T2428"): page((A1, "一"), following="p2"),
        next_url("p2"): page((A2, "二"), following="p3"),
    })
    with pytest.raises(ValueError, match="2 ページを超えました"):
        fetch_work("T2428", site.get, site.wait)


def test_fetch_work_refuses_lines_of_another_number():
    site = FakeSite({first_url("T2428"): page((A1, "一"), ("T2429_.77.0401c07", "混じった行"))})
    with pytest.raises(ValueError, match="別の番号の行"):
        fetch_work("T2428", site.get, site.wait)


def test_fetch_work_refuses_a_page_without_lines():
    site = FakeSite({first_url("T2428"): "<div>見つかりません</div>"})
    with pytest.raises(ValueError, match="行が取れない"):
        fetch_work("T2428", site.get, site.wait)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "evaluations" / "kukai").mkdir(parents=True)
    (tmp_path / fetch_sat.DEFAULT_EXCLUSIONS).write_text(EXCLUSIONS_YAML, encoding="utf-8")
    return tmp_path


def test_run_writes_one_work_with_the_source_and_lf_endings(repo: Path, capsys):
    site = FakeSite({first_url("T2429"): page(("T2429_.77.0401c05", "聲字實相義"), ("T2429_.77.0401c06", ""),
                                              ("T2429_.77.0401c07", "夫如來説法必藉文字"))})
    code = run(["--out-root", "knowledge/kukai", "--only", "T2429"], root=repo, get=site.get, wait=site.wait,
               today="2026-09-19")
    assert code == 0
    out = repo / "knowledge/kukai/primary/primary__声字実相義.md"
    raw = out.read_bytes()
    assert b"\r" not in raw
    text = raw.decode("utf-8")
    assert text.startswith("# 声字実相義（T2429）\n")
    assert f"出典：SAT 大蔵経テキストデータベース {first_url('T2429')}（2026-09-19 取得）" in text
    assert text.endswith("T2429_.77.0401c05 聲字實相義\nT2429_.77.0401c07 夫如來説法必藉文字\n")  # 空の行は除く
    assert site.asked == [first_url("T2429")]  # --only の 1 点だけ取りに行く
    assert "→ knowledge/kukai/primary/primary__声字実相義.md" in capsys.readouterr().out


def test_run_keeps_only_kukais_own_text_of_t2428(repo: Path, monkeypatch):
    last = "T2428_.77.0384a19"
    monkeypatch.setitem(fetch_sat.LAST_LINES, "T2428", (last, "空海の本文だけ"))
    site = FakeSite({first_url("T2428"): page((A1, "即身成佛義"), (last, "本文の終わり"),
                                              ("T2428_.77.0384a22", "眞言宗卽身成佛義問答一卷"))})
    assert run(["--out-root", "k", "--only", "T2428"], root=repo, get=site.get, wait=site.wait, today="2026-09-19") == 0
    text = (repo / "k/primary/primary__即身成仏義.md").read_text(encoding="utf-8")
    assert "- 範囲：空海の本文だけ" in text
    assert text.endswith(f"{last} 本文の終わり\n")
    assert "問答一卷" not in text


def test_run_stops_when_the_last_line_of_the_own_text_is_missing(repo: Path):
    site = FakeSite({first_url("T2428"): page((A1, "即身成佛義"))})
    with pytest.raises(ValueError, match="本文の最後の行"):
        run(["--out-root", "k", "--only", "T2428"], root=repo, get=site.get, wait=site.wait, today="2026-09-19")
    assert not (repo / "k/primary/primary__即身成仏義.md").exists()


def test_run_replaces_the_cut_range_with_the_marker(repo: Path):
    site = FakeSite({first_url("T2431"): page(("T2431_.77.0409b20", "前"), ("T2431_.77.0409b21", "去高雄舊居移入南山"),
                                              ("T2431_.77.0409b22", "吾性狎山水"), ("T2431_.77.0409b23", "後"))})
    assert run(["--out-root", "k", "--only", "T2431"], root=repo, get=site.get, wait=site.wait, today="2026-09-19") == 0
    text = (repo / "k/later-attribution/later-attribution__御遺告.md").read_text(encoding="utf-8")
    assert "区分：later-attribution（後世に空海の名で伝えられた文書。空海の言葉として引用しない）" in text
    assert text.endswith("T2431_.77.0409b20 前\n〔略 T2431_.77.0409b21–T2431_.77.0409b22〕\nT2431_.77.0409b23 後\n")
    assert "南山" not in text


def test_run_refuses_to_overwrite_an_existing_file(repo: Path, capsys):
    out = repo / "k/primary/primary__声字実相義.md"
    out.parent.mkdir(parents=True)
    out.write_text("前の版", encoding="utf-8")
    site = FakeSite({})
    assert run(["--out-root", "k", "--only", "T2429"], root=repo, get=site.get, wait=site.wait) == 1
    assert out.read_text(encoding="utf-8") == "前の版"
    assert site.asked == []  # 取りに行く前に止まる
    assert "上書きしない" in capsys.readouterr().err


def test_run_rejects_an_unknown_number(repo: Path, capsys):
    site = FakeSite({})
    assert run(["--out-root", "k", "--only", "T9999"], root=repo, get=site.get, wait=site.wait) == 2
    assert "取得する著作がありません" in capsys.readouterr().err


def test_run_fetches_all_nine_works_in_order_without_only(repo: Path):
    pages = {first_url(number): page((f"{number}{'' if number.endswith('A') else '_'}.00.0001a01", title))
             for number, title, _ in fetch_sat.WORKS}
    pages[first_url("T2428")] = page((A1, "即身成佛義"), (fetch_sat.LAST_LINES["T2428"][0], "終"))
    pages[first_url("T2431")] = page(*[(f"T2431_.77.0409b{n}", "行") for n in (20, 21, 22, 23)])
    site = FakeSite(pages)
    assert run(["--out-root", "k"], root=repo, get=site.get, wait=site.wait, today="2026-09-19") == 0
    assert site.asked == [first_url(number) for number, _, _ in fetch_sat.WORKS]
    written = sorted(p.relative_to(repo / "k").as_posix() for p in (repo / "k").rglob("*.md"))
    assert written == sorted(f"{r}/{r}__{t}.md" for _, t, r in fetch_sat.WORKS)
