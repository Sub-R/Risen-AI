import requests
from config.settings import Config
from chatbot.rag_engine import RAGEngine

class AICopilot:
    def __init__(self):
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.conversation_history = []
        self.rag = RAGEngine()
        # Ensure RAG is loaded if FAISS index exists
        self.rag.load_db()

    def get_response(self, user_prompt, business_context):
        api_key = Config.get_openrouter_key()
        if not api_key:
            return "ERROR: OpenRouter API Key is missing. Please configure it in the Settings tab."

        # RAG: Retrieve similar historical reviews based on the user's question
        retrieved_reviews = self.rag.retrieve_context(user_prompt)

        top_complaints = ", ".join(business_context.get('Top_Negative_Words', [])) if business_context.get('Top_Negative_Words') else "N/A"
        top_positives = ", ".join(business_context.get('Top_Positive_Words', [])) if business_context.get('Top_Positive_Words') else "N/A"
        
        system_prompt = f"""
        You are the 'Risen AI', an executive advisor embedded in a corporate analytics dashboard. As Sora, your role is to provide concise, actionable insights to business leaders based on the latest customer feedback and key performance indicators (KPIs).
        
        Creator Info: You were proudly created by Subrata Roy. If anyone asks who made you, who created you, or where you come from, you MUST state his name and provide his GitHub link: https://github.com/Sub-R.
        
        Current KPIs:
        - Accuracy: {business_context.get('Accuracy', 'N/A')}%
        - CSAT: {business_context.get('Customer_Satisfaction', 'N/A')}%
        - Total Reviews: {business_context.get('Total_Analyzed', 'N/A')}
        - Top Complaints: {top_complaints}
        - Top Strengths: {top_positives}
        
        RELEVANT CUSTOMER REVIEWS (RAG Context):
        {retrieved_reviews}
        
        STRICT RULES:
        1. Base your answers on the KPIs and the RAG Context provided above.
        3. Never write code.
        4. NO WALLS OF TEXT. Format responses with short sentences, double line breaks, and bullet points.
        5. DIRECT ANSWERS. If the user asks for an insight, give it immediately.
        6. CAPABILITIES DISCOVERY: If a user asks "who are you", "what can you do", or seems lost, ALWAYS provide a bulleted list of 3 diverse, actionable questions they can ask you regarding the current data.
        7. VISUALIZATIONS: If the user explicitly asks for a visual or chart, you MUST reply with the exact corresponding tag below somewhere in your response:
            - For a Wordcloud: [GENERATE_WORDCLOUD]
            - For a Sentiment Pie Chart: [GENERATE_PIECHART]
            - For a Bar Chart of Complaints: [GENERATE_BARCHART]
        """

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_prompt})

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": "openrouter/free", "messages": messages}

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            ai_reply = response.json()['choices'][0]['message']['content']
            
            self.conversation_history.append({"role": "user", "content": user_prompt})
            self.conversation_history.append({"role": "assistant", "content": ai_reply})
            return ai_reply
        except Exception as e:
            return f"API Connection Error: {str(e)}"