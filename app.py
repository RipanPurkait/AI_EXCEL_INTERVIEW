import os
from typing import Dict, TypedDict, Annotated, List, Literal
import operator
import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import logging
from dotenv import load_dotenv
load_dotenv()
# --- Configuration: Set Your Groq API Key ---
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize LLM ---
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    max_tokens=1024,
    timeout=10,
    max_retries=2,
)


# 1. Define State Structure

class InterviewState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], operator.add]
    current_node: str
    skills_assessed: Dict[str, float]
    total_questions: int
    correct_answers: int
    misunderstandings: List[str]
    confidence_level: Literal["Beginner", "Intermediate", "Advanced"]
    interview_complete: bool
    final_feedback: str
    asked_questions: List[str]
    current_question: Dict[str, str]


# 2. Generate Single Question

def generate_question(level: str, asked_questions: List[str]) -> Dict[str, str]:
    """Generate one unique Excel question for the specified level using LLM."""
    asked_prompt = f"Avoid these questions: {', '.join(asked_questions) or 'None'}"
    prompt = f"""
You are an expert Excel interviewer. Generate one unique Excel interview question for {level} level.
Return *only* a valid JSON object: {{"question": "", "answer": "", "rubric": "", "skill": "", "level": "{level}"}}
- Ensure the question is distinct, specific, and tests a unique Excel skill (e.g., functions, PivotTables, VBA, data analysis, charting).
- Do not repeat or rephrase any of these: {asked_prompt}.
- Provide a clear answer and rubric for evaluation.
Example: {{"question": "How do you use the IF function in Excel?", "answer": "Performs conditional logic to return values based on a condition.", "rubric": "Must mention conditional logic and return values.", "skill": "logical_functions", "level": "Beginner"}}
"""
    for attempt in range(5):  # Increased retries for robustness
        try:
            response = llm.invoke([SystemMessage(content=prompt)])
            if not response.content:
                logger.warning("Empty LLM response on attempt %d", attempt + 1)
                continue
            # Ensure response is valid JSON
            question = json.loads(response.content)
            if not all(key in question for key in ["question", "answer", "rubric", "skill", "level"]):
                logger.warning("Invalid JSON structure on attempt %d", attempt + 1)
                continue
            if question["question"] in asked_questions:
                logger.warning("Generated duplicate question on attempt %d", attempt + 1)
                continue
            logger.info("Generated question: %s", question["question"])
            return question
        except json.JSONDecodeError as e:
            logger.error("JSON parsing error on attempt %d: %s", attempt + 1, e)
        except Exception as e:
            logger.error("Failed to generate question on attempt %d: %s", attempt + 1, e)
    raise Exception("Failed to generate a unique question after 5 attempts")


# 3. Node Functions

def greeting_node(state: InterviewState) -> Dict:
    """Initial greeting and interview setup."""
    system_prompt = """
You are ExcelBot, an AI interviewer assessing Excel skills.
Introduce yourself warmly, explain this is a mock technical interview with exactly 5 questions,
and ask if the candidate is ready to begin. Keep it concise and friendly.
"""
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt)])
    chain = prompt | llm | StrOutputParser()
    try:
        response = chain.invoke({})
    except Exception as e:
        logger.error("Greeting node error: %s", e)
        response = "Hi! I'm ExcelBot, here to test your Excel skills with 5 questions. Ready to start?"
    
    return {
        "messages": [AIMessage(content=response)],
        "current_node": "warm_up",
        "skills_assessed": {},
        "total_questions": 0,
        "correct_answers": 0,
        "misunderstandings": [],
        "confidence_level": "Beginner",
        "interview_complete": False,
        "final_feedback": "",
        "asked_questions": [],
        "current_question": {}
    }

def warm_up_node(state: InterviewState) -> Dict:
    """Ask a beginner-level question."""
    if state["total_questions"] >= 5:
        return wrap_up_node(state)
    question = generate_question("Beginner", state["asked_questions"])
    return {
        "messages": [AIMessage(content=question["question"])],
        "current_node": "evaluate_warmup",
        "asked_questions": state["asked_questions"] + [question["question"]],
        "current_question": question
    }

def core_technical_node(state: InterviewState) -> Dict:
    """Ask an intermediate-level question."""
    if state["total_questions"] >= 5:
        return wrap_up_node(state)
    question = generate_question("Intermediate", state["asked_questions"])
    return {
        "messages": [AIMessage(content=question["question"])],
        "current_node": "evaluate_core",
        "asked_questions": state["asked_questions"] + [question["question"]],
        "current_question": question
    }

def scenario_node(state: InterviewState) -> Dict:
    """Ask an advanced scenario-based question."""
    if state["total_questions"] >= 5:
        return wrap_up_node(state)
    question = generate_question("Advanced", state["asked_questions"])
    return {
        "messages": [AIMessage(content=question["question"])],
        "current_node": "evaluate_scenario",
        "asked_questions": state["asked_questions"] + [question["question"]],
        "current_question": question
    }

def follow_up_node(state: InterviewState) -> Dict:
    """Ask a follow-up to clarify misunderstandings."""
    human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    last_answer = human_msgs[-1].content if human_msgs else "No answer provided."
    
    prompt = f"""
The candidate answered: "{last_answer}"
They seem confused about: "{state['current_question']['question']}"
Ask a gentle follow-up question to clarify their understanding.
Be encouraging and specific. Example: "Can you explain how you’d handle errors in VLOOKUP?"
"""
    try:
        response = llm.invoke([SystemMessage(content=prompt)])
        return {
            "messages": [AIMessage(content=response.content)],
            "current_node": "evaluate_followup",
            "asked_questions": state["asked_questions"],
            "current_question": state["current_question"]
        }
    except Exception as e:
        logger.error("Follow-up node error: %s", e)
        return {
            "messages": [AIMessage(content="Could you clarify your last answer a bit more?")],
            "current_node": "evaluate_followup",
            "asked_questions": state["asked_questions"],
            "current_question": state["current_question"]
        }

def candidate_qa_node(state: InterviewState) -> Dict:
    """Ask a final question (5th question)."""
    if state["total_questions"] >= 5:
        return wrap_up_node(state)
    question = generate_question("Advanced", state["asked_questions"])
    return {
        "messages": [AIMessage(content=question["question"])],
        "current_node": "evaluate_final",
        "asked_questions": state["asked_questions"] + [question["question"]],
        "current_question": question
    }

# -------------------------------
# 4. Evaluation Logic
# -------------------------------
def evaluate_answer(state: InterviewState, next_correct: str, next_incorrect: str) -> Dict:
    """Evaluate the user's answer using LLM."""
    question = state["current_question"]
    rubric = question.get("rubric", "")
    skill_name = question.get("skill", "")
    
    user_msg = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    ai_msg = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    
    if not user_msg or not ai_msg:
        logger.warning("No valid answer to evaluate")
        return {
            "current_node": next_incorrect,
            "asked_questions": state["asked_questions"],
            "current_question": state["current_question"],
            "total_questions": state["total_questions"] + 1,
            "correct_answers": state["correct_answers"],
            "skills_assessed": state["skills_assessed"],
            "misunderstandings": state["misunderstandings"] + [ai_msg.content[:100] + "..."] if ai_msg else state["misunderstandings"]
        }

    eval_prompt = f"""
Evaluate:
Question: "{ai_msg.content}"
Answer: "{user_msg.content}"
Rubric: {rubric}
Rate: CORRECT / PARTIAL / INCORRECT only.
Provide a brief explanation (20-30 words).
Return *only* a valid JSON object: {{"rating": "", "explanation": ""}}
Example: {{"rating": "CORRECT", "explanation": "Answer correctly describes conditional logic and return values."}}
"""
    try:
        response = llm.invoke([SystemMessage(content=eval_prompt)])
        result = json.loads(response.content)
        rating = result["rating"].upper()
        explanation = result["explanation"]
    except Exception as e:
        logger.error("Evaluation error: %s", e)
        rating = "INCORRECT"
        explanation = "Failed to evaluate answer due to processing error."
    
    new_total = state["total_questions"] + 1
    new_correct = state["correct_answers"] + (1 if rating == "CORRECT" else 0)
    score_map = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}
    
    new_skills = state["skills_assessed"].copy()
    new_misunderstandings = state["misunderstandings"].copy()
    
    if skill_name:
        new_skills[skill_name] = score_map.get(rating, 0.0)
    if rating == "INCORRECT":
        new_misunderstandings.append(ai_msg.content[:100] + "...")
    
    next_node = next_correct if rating in ["CORRECT", "PARTIAL"] else next_incorrect
    return {
        "current_node": next_node,
        "total_questions": new_total,
        "correct_answers": new_correct,
        "skills_assessed": new_skills,
        "misunderstandings": new_misunderstandings,
        "asked_questions": state["asked_questions"],
        "current_question": state["current_question"]
    }

def evaluate_warmup(state: InterviewState) -> Dict:
    return evaluate_answer(state, "core_technical", "core_technical")

def evaluate_core(state: InterviewState) -> Dict:
    return evaluate_answer(state, "core_technical2", "follow_up")

def evaluate_core2(state: InterviewState) -> Dict:
    return evaluate_answer(state, "scenario", "follow_up")

def evaluate_scenario(state: InterviewState) -> Dict:
    return evaluate_answer(state, "candidate_qa", "follow_up")

def evaluate_final(state: InterviewState) -> Dict:
    if state["total_questions"] >= 5:
        return wrap_up_node(state)
    return evaluate_answer(state, "wrap_up", "wrap_up")

def evaluate_followup(state: InterviewState) -> Dict:
    return {
        "current_node": "scenario" if state["total_questions"] < 4 else "candidate_qa",
        "asked_questions": state["asked_questions"],
        "current_question": state["current_question"]
    }

# -------------------------------
# 5. Wrap-Up Node
# -------------------------------
def wrap_up_node(state: InterviewState) -> Dict:
    """Generate final feedback after 5 questions."""
    accuracy = state["correct_answers"] / max(1, state["total_questions"])
    level = "Advanced" if accuracy > 0.7 else "Intermediate" if accuracy > 0.4 else "Beginner"
    
    skills_summary = "; ".join([f"{k}: {v*100:.0f}%" for k, v in state["skills_assessed"].items()])
    misunderstandings = "; ".join(state["misunderstandings"]) or "None"
    
    feedback_prompt = f"""
Write a 100-120 word feedback summary:
- Level: {level}
- Skills assessed: {skills_summary}
- Misunderstandings: {misunderstandings}
- Strengths: Highlight strong areas based on skills.
- Areas to improve: Suggest focus based on misunderstandings.
- Practical tip: One actionable Excel tip (e.g., learn a specific function).
Tone: Mentor-like, kind, constructive.
Return *only* the feedback text.
"""
    try:
        response = llm.invoke([SystemMessage(content=feedback_prompt)])
        feedback = response.content
    except Exception as e:
        logger.error("Feedback generation error: %s", e)
        feedback = f"You're at {level} level. Strengths: {skills_summary}. Improve on {misunderstandings}. Tip: Practice dynamic ranges with OFFSET."
    
    return {
        "final_feedback": feedback,
        "interview_complete": True,
        "current_node": "END",
        "messages": [*state["messages"], AIMessage(content=feedback)],
        "asked_questions": state["asked_questions"],
        "current_question": state["current_question"]
    }

# -------------------------------
# 6. Build LangGraph
# -------------------------------
workflow = StateGraph(InterviewState)
workflow.set_entry_point("greeting")

# Add nodes
nodes = [
    ("greeting", greeting_node),
    ("warm_up", warm_up_node),
    ("evaluate_warmup", evaluate_warmup),
    ("core_technical", core_technical_node),
    ("evaluate_core", evaluate_core),
    ("core_technical2", core_technical_node), 
    ("evaluate_core2", evaluate_core2),
    ("scenario", scenario_node),
    ("evaluate_scenario", evaluate_scenario),
    ("candidate_qa", candidate_qa_node),
    ("evaluate_final", evaluate_final),
    ("follow_up", follow_up_node),
    ("evaluate_followup", evaluate_followup),
    ("wrap_up", wrap_up_node),
]
for node_name, func in nodes:
    workflow.add_node(node_name, func)

# Define edges
workflow.add_edge("greeting", "warm_up")
workflow.add_edge("warm_up", "evaluate_warmup")
workflow.add_edge("core_technical", "evaluate_core")
workflow.add_conditional_edges("evaluate_core", lambda x: x["current_node"])
workflow.add_edge("core_technical2", "evaluate_core2")
workflow.add_conditional_edges("evaluate_core2", lambda x: x["current_node"])
workflow.add_edge("scenario", "evaluate_scenario")
workflow.add_conditional_edges("evaluate_scenario", lambda x: x["current_node"])
workflow.add_edge("candidate_qa", "evaluate_final")
workflow.add_conditional_edges("evaluate_final", lambda x: x["current_node"])
workflow.add_edge("follow_up", "evaluate_followup")
workflow.add_conditional_edges("evaluate_followup", lambda x: x["current_node"])
workflow.add_edge("wrap_up", END)

app = workflow.compile()

# -------------------------------
# 7. Streamlit Interface
# -------------------------------
def main():
    """Main Streamlit app."""
    st.set_page_config(page_title="📊 AI Excel Mock Interviewer", page_icon="📊")
    st.title("📊 AI Excel Mock Interviewer")
    st.markdown("""
    Welcome! I'm ExcelBot, your AI-powered interviewer for Excel skills.
    We'll go through 5 questions on functions, formulas, and scenarios.
    Type 'start' to begin or 'restart' to reset!
    """)

    # Initialize session state
    if "chat_state" not in st.session_state:
        st.session_state.chat_state = {
            "messages": [],
            "current_node": "greeting",
            "skills_assessed": {},
            "total_questions": 0,
            "correct_answers": 0,
            "misunderstandings": [],
            "confidence_level": "Beginner",
            "interview_complete": False,
            "final_feedback": "",
            "asked_questions": [],
            "current_question": {}
        }
    if "history" not in st.session_state:
        st.session_state.history = []

    # Display chat history
    for user_msg, bot_msg in st.session_state.history:
        with st.chat_message("user"):
            if user_msg:
                st.write(user_msg)
        with st.chat_message("assistant"):
            st.write(bot_msg)

    # Input form
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Your response:", placeholder="Type your answer here...")
        col1, col2 = st.columns(2)
        with col1:
            submit_button = st.form_submit_button("Send")
        with col2:
            restart_button = st.form_submit_button("Restart Interview")

    def respond(message: str, chat_state: Dict) -> tuple[str, Dict]:
        """Handle user input and return bot response with updated state."""
        if not message:
            return "Please type a response to continue.", chat_state

        # Handle restart
        if message.lower() == "restart":
            chat_state = {
                "messages": [],
                "current_node": "greeting",
                "skills_assessed": {},
                "total_questions": 0,
                "correct_answers": 0,
                "misunderstandings": [],
                "confidence_level": "Beginner",
                "interview_complete": False,
                "final_feedback": "",
                "asked_questions": [],
                "current_question": {}
            }
            try:
                result = app.invoke(chat_state, {"recursion_limit": 50})
                bot_response = result["messages"][-1].content
                return bot_response, result
            except Exception as e:
                logger.error("Graph execution error on restart: %s", e)
                return "Error restarting. Please try again.", chat_state

        # Add user message
        if message.lower() != "start" or chat_state["current_node"] != "greeting":
            chat_state["messages"].append(HumanMessage(content=message))

        try:
            result = app.invoke(chat_state, {"recursion_limit": 50})
            bot_response = result["messages"][-1].content if result["messages"] else "..."
            return bot_response, result
        except Exception as e:
            logger.error("Graph execution error: %s", e)
            return "Error processing response. Please try again.", chat_state

    # Handle send
    if submit_button and user_input:
        bot_response, new_state = respond(user_input, st.session_state.chat_state)
        st.session_state.history.append((user_input, bot_response))
        st.session_state.chat_state = new_state
        st.rerun()

    # Handle restart
    if restart_button:
        st.session_state.chat_state = {
            "messages": [],
            "current_node": "greeting",
            "skills_assessed": {},
            "total_questions": 0,
            "correct_answers": 0,
            "misunderstandings": [],
            "confidence_level": "Beginner",
            "interview_complete": False,
            "final_feedback": "",
            "asked_questions": [],
            "current_question": {}
        }
        st.session_state.history = []
        st.rerun()

if __name__ == "__main__":
    main()
