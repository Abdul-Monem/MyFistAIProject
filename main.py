import math
import sqlite3
from openai import OpenAI

# ── 1. YOUR STRICT SAFETY SYSTEM PROMPT ──
SYSTEM_PROMPT = """You are a local, offline support agent
for gas field inspection and maintenance engineers. Behaviour Rules:
– Always prioritise safety. If a procedure involves risk, explicitly call it out.
– Do not hallucinate procedures, measurements, or tolerances.
– If the answer is not in the provided context, say:
“This information is not available in the local knowledge base.”
Response Format:
– Summary (1-2 lines)
– Safety Warnings (if applicable)
– Step-by-step Guidance
– Reference (document name + section)"""

# ── 2. SIMPLE TF-IDF LOGIC FOR RETRIEVAL ──
def term_frequency(text):
    words = text.lower().split()
    tf = {}
    for word in words:
        tf[word] = tf.get(word, 0) + 1
    return tf

def cosine_similarity(tf1, tf2):
    intersection = set(tf1.keys()) & set(tf2.keys())
    dot_product = sum(tf1[x] * tf2[x] for x in intersection)
    
    sum1 = sum(tf1[x]**2 for x in tf1.keys())
    sum2 = sum(tf2[x]**2 for x in tf2.keys())
    
    if not sum1 or not sum2:
        return 0.0
    return dot_product / (math.sqrt(sum1) * math.sqrt(sum2))

# ── 3. FETCH DATA FROM YOUR EXISTING SQLITE DB ──
# ── 3. FETCH DATA FROM YOUR EXISTING SQLITE DB ──
def retrieve_context(query, top_k=2):
    query_tf = term_frequency(query)
    scored_chunks = []

    # 1. Connect to your active database file
    conn = sqlite3.connect('data/rag.db')
    cursor = conn.cursor()
    
    # 2. Execute the query using your exact table columns
    cursor.execute("SELECT id, content, title FROM chunks")
    
    # 3. This line automatically downloads all matching text data into memory
    rows = cursor.fetchall()
    
    # 4. Safely disconnect from the database file
    conn.close()

    # 5. Loop through the data python pulled from the database
    for row_id, content, title in rows:
        chunk_tf = term_frequency(content)
        score = cosine_similarity(query_tf, chunk_tf)
        if score > 0:
            scored_chunks.append({"text": content, "doc": title, "score": score})
            
    # 6. Rank them from best match to worst match
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    return scored_chunks[:top_k]

# ── 4. MAIN INTERACTIVE EXECUTION LOOP ──
def main():
    print("=== Gas Field RAG – Python Local Support Agent ===")
    
    # Directly target your active background Foundry server on port 
    chat_client = OpenAI(
        base_url="http://127.0.0.1:44005/v1",
        api_key="nokey"
    )
    
    print("\nSystem ready for Python execution. Connected to port 44005.")
    print('Type "quit" to exit.\n')

    while True:
        query = input("Question: ").strip()
        if not query or query.lower() == "quit":
            break

        # Step A: Retrieve localized data chunks matching the keywords
        results = retrieve_context(query, top_k=2)
        
        if not results:
            context = "No relevant local documents found matching these keywords."
        else:
            context = "\n".join(f"[Doc: {res['doc']}] {res['text']}" for res in results)

        # Step B: Build messages array embedding the exact documentation context
        messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nProvided Local Context:\n{context}"},
            {"role": "user", "content": query}
        ]

        print("\nAnswer: ", end="", flush=True)
        try:
            # Step C: Stream the generation directly via Python client safely
            response = chat_client.chat.completions.create(
                model="phi-3.5-mini",
                messages=messages,
                temperature=0.1,
                stream=True
            )
            
            for chunk in response:
                # FIX: Verify choices list is not empty before parsing
                if chunk.choices and len(chunk.choices) > 0:
                    content = chunk.choices[0].delta.content
                    if content:
                        print(content, end="", flush=True)
            print("\n")
            
        except Exception as e:
            print(f"\n[Error] Failed to connect to local worker daemon: {e}\n")

if __name__ == "__main__":
    main()