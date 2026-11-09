from pathlib import Path
from streamlit.testing.v1 import AppTest
app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py')).run(timeout=60)
assert not app.exception, [e.message for e in app.exception]
assert app.metric[1].value == '4,595'
for features in [[], ['DEBT'], ['DEBT','HOUSES'], ['ASSET','INCOME','NETWORTH']]:
    app.multiselect[0].set_value(features).run(timeout=60)
    assert not app.exception, [e.message for e in app.exception]
    if len(features)<2: assert len(app.warning)>0
for k in [2,8]:
    app.slider[0].set_value(k).run(timeout=60)
    assert not app.exception, [e.message for e in app.exception]
print('Dashboard default, empty/single/multiple features, and k endpoints passed.')
