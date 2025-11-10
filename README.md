# 🎧 Vibe Matcher — AI Mini Recommendation System  
Prototype Submission for Nexora AI Internship

## 📌 Overview
Vibe Matcher is a lightweight AI-powered recommendation prototype that matches a user’s **vibe query** (e.g., “aesthetic streetwear”, “soft pastel college look”) with the most relevant fashion items using **semantic embeddings** and **cosine similarity**.  
This project demonstrates how modern embedding models can be used to build fast, scalable, real-world recommendation systems.

The notebook includes:  
✅ Embedding generation using OpenAI (or compatible embedding model)  
✅ A small fashion dataset (5–10 items with descriptions)  
✅ Cosine similarity–based ranking  
✅ Top-3 vibe match results  
✅ Evaluation metrics (latency, consistency checks)  
✅ Reflection section  
✅ Clean, modular, reproducible code  

---

## 🚀 Features
- **Vibe Query → Embedding → Similarity Search**  
- **Product embeddings stored in DataFrame**  
- **Efficient cosine similarity implementation**  
- **Ranking visualization**  
- **Latency measurement & accuracy checks**  
- **Reflection on system performance**

---

## 📂 Project Structure
vibe-matcher/
│

├── 
vibe_matcher.ipynb # Main notebook (with outputs)

├── data/

│ 
└── products.csv # Sample fashion dataset (5–10 items)

├
── src/

│
├── embeddings.py # Embedding utilities

│ 
├── similarity.py # Cosine similarity + ranking

│
└── plotting.py # Visualization utilities
└── README.md 



---



## ⚙️ Requirements
Install the required Python packages in Colab or locally:

```bash
pip install numpy pandas scikit-learn matplotlib openai

📝 How to Run

Open vibe_matcher.ipynb in Google Colab or Jupyter.

Upload the dataset products.csv.

Run all notebook cells to generate outputs.

Enter your vibe query when prompted.

View the top-3 recommended products along with:

Product names

Descriptions

Similarity scores

Ranked results

---

**## ✨ Sample Vibe Query:**

"Minimal Korean streetwear with clean lines"

---

**🧪 Evaluation**

The prototype includes simple evaluation steps:

Latency Test: Measure time for embedding generation and similarity computation.

Embedding Consistency: Verify repeated queries produce stable similarity scores.

Qualitative Accuracy: Check if top matches are relevant to the input vibe.

Reflection: Summarize what worked and potential improvements.

---

**💭 Reflection Summary**

Works well for short, descriptive vibe queries and small datasets.

Can scale to larger datasets with:

Fine-tuned embeddings

Additional metadata (colors, tags, styles)

Possible improvements:

Integration with vector databases (e.g., Pinecone) for faster search

Handling ambiguous or rare vibe queries with fallback strategies
---

**📜 License**

This project is for educational and submission purposes only.

