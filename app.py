import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Groq LLM setup
llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.3-70b-versatile"
)

def clean_sql(query):
    query = query.strip()
    query = query.replace("```sql", "").replace("```", "")
    return query.strip()

# ── Engine ─────────────────────────────────────────────
DB_URL = "sqlite:///data.db"
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

# ── History table ──────────────────────────────────────
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS query_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            table_name TEXT,
            asked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """))
    conn.commit()

def save_question(question, table_name):
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO query_history (question, table_name) VALUES (:q, :t)"),
            {"q": question, "t": table_name}
        )
        conn.commit()

def load_history(table_name):
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT question, asked_at FROM query_history WHERE table_name = :t ORDER BY asked_at DESC"),
            {"t": table_name}
        )
        return result.fetchall()

def clear_history(table_name):
    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM query_history WHERE table_name = :t"),
            {"t": table_name}
        )
        conn.commit()

# ── UI ─────────────────────────────────────────────────
st.title("📊 Sales Chatbot")

# ── Multiple CSV Upload ────────────────────────────────
uploaded_files = st.sidebar.file_uploader(
    "📂 Upload CSV files",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:
    table_names = []

    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)
        table_name = uploaded_file.name.replace(".csv", "").replace(" ", "_").lower()
        table_names.append(table_name)
        df.to_sql(table_name, engine, if_exists="replace", index=False)

        st.sidebar.success(f"✅ '{table_name}' uploaded!")
        st.sidebar.write(f"**Columns:** {list(df.columns)}")
        st.sidebar.write(f"**Rows:** {len(df)}")
        st.sidebar.markdown("---")

    # ── Query History Sidebar ──────────────────────────
    current_tables = ", ".join(table_names)
    st.sidebar.subheader("🕘 Query History")
    history = load_history(current_tables)
    if history:
        for i, row in enumerate(history):
            st.sidebar.markdown(f"**{i+1}.** {row[0]}")
            st.sidebar.caption(f"🕐 {row[1]}")
        if st.sidebar.button("🗑️ Clear History"):
            clear_history(current_tables)
            st.rerun()
    else:
        st.sidebar.info("No questions asked yet!")

    # ── Database connect karo ──────────────────────────
    db = SQLDatabase.from_uri(
        DB_URL,
        include_tables=table_names
    )

    def get_limited_schema():
        schema = db.get_table_info()
        return schema[:3000]

    # SQL prompt
    sql_prompt = ChatPromptTemplate.from_template("""
You are a SQL expert. Look at the schema below and write only the SQL query.
Do not explain anything, just write the query. No backticks. No square brackets.
IMPORTANT: Column names with spaces must always be wrapped in double quotes like "Review Rating".
IMPORTANT: Aggregate functions must use round brackets like MAX("Review Rating"), MIN("Age"), SUM("Purchase Amount (USD)"), COUNT(*).
You can use JOIN if the question needs data from multiple tables.
Available tables: {tables}

Schema: {schema}
Question: {question}
SQL Query:
""")

    # Answer prompt
    answer_prompt = ChatPromptTemplate.from_template("""
Question: {question}
Result: {result}

Give a simple one sentence answer in English only.
""")

    # Chart prompt
    chart_prompt = ChatPromptTemplate.from_template("""
You are a data visualization expert.
I have this data result from a SQL query: {result}
The columns are: {columns}

Which chart type is best to visualize this?
Reply with ONLY one word: bar, line, pie, or none.
""")

    # Insight prompt
    insight_prompt = ChatPromptTemplate.from_template("""
You are a senior data analyst. Analyze this dataset and give 5 key business insights.
Be specific, use numbers from the data.

Dataset columns: {columns}
Dataset sample: {sample}

Give exactly 5 insights in simple English. Number them 1 to 5.
""")

    # ── Chains ────────────────────────────────────────
    sql_chain = (
        RunnablePassthrough.assign(schema=lambda _: get_limited_schema())
        | RunnablePassthrough.assign(tables=lambda _: ", ".join(table_names))
        | sql_prompt
        | llm
        | StrOutputParser()
        | clean_sql
    )

    def get_limited_result(x):
        result = db.run(x["query"])
        return str(result)[:2000]

    full_chain = (
        RunnablePassthrough.assign(query=sql_chain)
        .assign(result=get_limited_result)
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    def get_chart_type(result, columns):
        chain = chart_prompt | llm | StrOutputParser()
        return chain.invoke({"result": result, "columns": str(columns)}).strip().lower()

    def get_sql_result_df(question):
        sql = clean_sql(sql_chain.invoke({"question": question}))
        with engine.connect() as conn:
            return pd.read_sql(sql, conn).head(100), sql

    def show_chart(df_result, question=""):
        if df_result is None or df_result.empty or len(df_result.columns) < 2:
            return
        cols = list(df_result.columns)
        q_lower = question.lower()

        if "bar" in q_lower:
            chart_type = "bar"
        elif "line" in q_lower:
            chart_type = "line"
        elif "pie" in q_lower:
            chart_type = "pie"
        else:
            chart_type = get_chart_type(df_result.to_string(), cols)

        fig, ax = plt.subplots()

        if chart_type == "bar":
            ax.bar(df_result[cols[0]], df_result[cols[1]], color="steelblue")
            ax.set_xlabel(cols[0])
            ax.set_ylabel(cols[1])
            ax.set_title("📊 Bar Chart")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
        elif chart_type == "line":
            ax.plot(df_result[cols[0]], df_result[cols[1]], color="green", marker="o")
            ax.set_xlabel(cols[0])
            ax.set_ylabel(cols[1])
            ax.set_title("📈 Line Chart")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)
        elif chart_type == "pie":
            ax.pie(df_result[cols[1]], labels=df_result[cols[0]], autopct="%1.1f%%")
            ax.set_title("🥧 Pie Chart")
            st.pyplot(fig)

        plt.close(fig)

    # ── Data Preview ───────────────────────────────────
    st.subheader("📋 Data Preview")
    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
        df_preview = pd.read_csv(uploaded_file)
        table_name = uploaded_file.name.replace(".csv", "").replace(" ", "_").lower()
        with st.expander(f"📄 {table_name}"):
            st.dataframe(df_preview.head())

            if st.button(f"🔍 Generate Insights — {table_name}"):
                with st.spinner("Analyzing..."):
                    try:
                        chain = insight_prompt | llm | StrOutputParser()
                        insights = chain.invoke({
                            "columns": list(df_preview.columns),
                            "sample": df_preview.head(30).to_string()
                        })
                        st.subheader("💡 Key Insights")
                        st.write(insights)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    # ── Chat ───────────────────────────────────────────
    st.subheader("💬 Ask Questions")
    st.caption(f"Available tables: {', '.join(table_names)}")

    question = st.chat_input("Ask anything about your data...")

    if question:
        with st.spinner("Thinking..."):
            try:
                save_question(question, current_tables)

                answer = full_chain.invoke({"question": question})
                st.chat_message("user").write(question)
                st.chat_message("assistant").write(answer)

                df_result, sql = get_sql_result_df(question)
                if df_result is not None and not df_result.empty:
                    st.dataframe(df_result)
                    show_chart(df_result, question)

            except Exception as e:
                st.error(f"Error: {str(e)}")

else:
    st.info("👈 Upload one or more CSV files from the sidebar to get started!")