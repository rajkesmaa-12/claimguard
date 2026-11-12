"""Streamlit demo:  streamlit run app.py"""
import streamlit as st

from claimguard.evidence import WikipediaSource
from claimguard.explain import explain
from claimguard.nli import TransformersNLI
from claimguard.verify import REFUTED, SUPPORTED, ClaimGuard

st.set_page_config(page_title="ClaimGuard", page_icon="🛡️")
st.title("🛡️ ClaimGuard — check a claim against Wikipedia")


@st.cache_resource
def load_guard():
    return ClaimGuard(WikipediaSource(), TransformersNLI())


claim = st.text_input("Claim", "The Eiffel Tower is located in Berlin.")
if st.button("Check") and claim:
    with st.spinner("Retrieving evidence and running NLI…"):
        v = load_guard().check(claim)
    colour = {SUPPORTED: "green", REFUTED: "red"}.get(v.label, "orange")
    st.markdown(f"### :{colour}[{v.label}]  ·  confidence {v.confidence:.2f}")
    if (why := explain(v)):
        st.info(why)
    for e in v.evidence:
        st.markdown(f"> {e.evidence.text}\n\nentail **{e.entailment:.2f}** · contra **{e.contradiction:.2f}** · [source]({e.evidence.source})")
