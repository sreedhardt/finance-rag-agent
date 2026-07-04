from finance_rag.tools import TOOL_DECLARATIONS, openai_tool_declarations


def test_openai_declarations_mirror_gemini_source():
    converted = openai_tool_declarations()
    assert len(converted) == len(TOOL_DECLARATIONS)
    by_name = {t["function"]["name"]: t["function"] for t in converted}
    for decl in TOOL_DECLARATIONS:
        fn = by_name[decl.name]
        assert fn["description"] == decl.description
        params = fn["parameters"]
        assert params["type"] == "object"
        assert set(params["required"]) == set(decl.parameters.required)
        # every property survives conversion with a lowercase JSON-schema type
        for prop_name, prop_schema in decl.parameters.properties.items():
            assert params["properties"][prop_name]["type"] == prop_schema.type.name.lower()


def test_sql_tool_schema_is_valid_openai_format():
    (sql_tool,) = [t for t in openai_tool_declarations()
                   if t["function"]["name"] == "query_financials"]
    assert sql_tool["type"] == "function"
    assert sql_tool["function"]["parameters"]["properties"]["sql"]["type"] == "string"
    assert "quarterly_financials" in sql_tool["function"]["description"]
