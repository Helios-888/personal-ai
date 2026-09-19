"""sat: SAT 大蔵経テキストデータベースの詳細ページから、行 ID つきの本文を取り出す部品。"""
from scripts.lib.sat import merge_lines, next_useid, parse_lines, render_markdown

PAGE = (
    '<div>header 空海</div>'
    '<span style="color:black">T2428_.77.0382c21: </span><a name="0382c21">業。是名羯<button class="ftntf2" '
    'title="羯＝竭＜甲＞">30</button>磨曼荼羅。是<button class="ftntf2" title="x">31</button>名四種曼荼羅。</a><br/>'
    '<table><tr><td><span style="color:black">T2428_.77.0382c22: </span></td><td><a name="0382c22">次明龍趣</td>'
    '<td><span style="font-size:8pt">亦是傍生趣攝</span></td><td></a></td></tr></table>'
    '<span style="color:black">T2428_.77.0382c23: </span><a name="0382c23">荼羅&amp;</a><br/></span>'
    '<div id="nextl" onclick="location.href=\'ddb-sat2.php?mode=detail&nonum=&kaeri=&mode2=2&useid=2428_,77,0383a01\'">'
    '</div><div>Footnote: 羯＝竭</div>'
)


def test_parse_lines_keeps_line_ids_and_drops_note_numbers():
    assert parse_lines(PAGE) == [
        ("T2428_.77.0382c21", "業。是名羯磨曼荼羅。是名四種曼荼羅。"),
        ("T2428_.77.0382c22", "次明龍趣（亦是傍生趣攝）"),  # 割注は括弧に入れる
        ("T2428_.77.0382c23", "荼羅&"),  # 最後の行の後ろの案内や注は入れない
    ]


def test_parse_lines_reads_ids_without_the_underscore():
    page = '<span style="color:black">T2203A.57.0011c08: </span><a name="0011c08">此經總有五分。</a><br/>'
    assert parse_lines(page) == [("T2203A.57.0011c08", "此經總有五分。")]


def test_next_useid_follows_the_continuation_link():
    assert next_useid(PAGE) == "2428_,77,0383a01"
    assert next_useid("<div>end</div>") is None


def test_merge_lines_drops_lines_already_seen_and_keeps_order():
    first = [("T1_.77.0001a01", "一"), ("T1_.77.0001a02", "二")]
    second = [("T1_.77.0001a01", "一"), ("T1_.77.0001a02", "二"), ("T1_.77.0001a03", "三")]
    assert merge_lines(first, second) == [*first, ("T1_.77.0001a03", "三")]


def test_merge_lines_refuses_the_same_id_with_different_text():
    import pytest

    with pytest.raises(ValueError, match="T1_.77.0001a01"):
        merge_lines([("T1_.77.0001a01", "一")], [("T1_.77.0001a01", "壱")])


def test_render_markdown_puts_the_source_above_the_numbered_lines():
    text = render_markdown(
        title="即身成仏義", number="T2428", reliability="primary", reliability_note="空海本人の著作",
        url="https://example/ddb-sat2.php?mode=detail&useid=2428_", fetched="2026-09-20",
        lines=[("T2428_.77.0381b16", "即身成佛義"), ("T2428_.77.0381b17", "問曰。")],
    )
    assert text.startswith("# 即身成仏義（T2428）\n")
    assert "区分：primary（空海本人の著作）" in text
    assert "T2428_.77.0381b16–T2428_.77.0381b17" in text
    assert text.endswith("T2428_.77.0381b16 即身成佛義\nT2428_.77.0381b17 問曰。\n")


def test_render_markdown_drops_empty_lines_and_states_the_scope():
    text = render_markdown(
        title="即身成仏義", number="T2428", reliability="primary", reliability_note="空海本人の著作",
        url="u", fetched="2026-09-20", scope_note="0384a22 以降の異本は入れない",
        lines=[("T2428_.77.0381b16", "即身成佛義"), ("T2428_.77.0381b17", ""), ("T2428_.77.0381b18", "問曰。")],
    )
    assert "- 範囲：0384a22 以降の異本は入れない" in text
    assert "T2428_.77.0381b17 " not in text
    assert text.endswith("T2428_.77.0381b16 即身成佛義\nT2428_.77.0381b18 問曰。\n")
