/**
 * Foundry Local chat engine.
 * Bypasses native C++ bindings to prevent Linux Segfaults by
 * connecting directly to the running background Foundry server.
 */
import { OpenAI } from "openai";
import { VectorStore } from "./vectorStore.js";
import { config } from "./config.js";
import { SYSTEM_PROMPT, SYSTEM_PROMPT_COMPACT } from "./prompts.js";

export class ChatEngine {
  constructor() {
    this.chatClient = null;
    this.model = null;
    this.store = null;
    this.compactMode = false;
    this.modelAlias = null;
    /** @type {(status: {phase: string, message: string, progress?: number}) => void} */
    this._statusCallback = null;
  }

  /** Register a callback that receives init status updates for the UI. */
  onStatus(callback) {
    this._statusCallback = callback;
  }

  _emitStatus(phase, message, progress) {
    const status = { phase, message, ...(progress !== undefined && { progress }) };
    console.log(`[ChatEngine] ${message}`);
    if (this._statusCallback) this._statusCallback(status);
  }

  /**
   * Initialize the engine: Bypasses native C++ launcher to avoid crash
   * and links directly to the background service at http://127.0.0.1:44565
   */
  async init() {
    this._emitStatus("init", "Initializing Foundry Local SDK (HTTP Client mode)...");

    // We hardcode the model details to bypass native hardware scanning
    this.modelAlias = config.model || "phi-3.5-mini";
    this._emitStatus("variant", `Using running background model instance: ${this.modelAlias}`);

    // Create a standard, crash-free HTTP OpenAI client pointing to your background server
    this.chatClient = new OpenAI({
      baseURL: "http://127.0.0.1:44005/v1/", // Using the active port from your server terminal!
      apiKey: "nokey" // No key required for local server
    });

    this._emitStatus("ready", `Model ready via background service connection.`);

    // Open the local vector store
    this.store = new VectorStore(config.dbPath);
    const count = this.store.count();
    this._emitStatus("ready", `Vector store ready: ${count} chunks indexed.`);

    if (count === 0) {
      console.warn("[ChatEngine] WARNING: No documents ingested. Run 'npm run ingest' first.");
    }
  }

  /** Expose the vector store for direct operations (e.g. upload ingestion). */
  getStore() {
    return this.store;
  }

  /**
   * Set compact mode for extreme latency / edge devices.
   */
  setCompactMode(enabled) {
    this.compactMode = enabled;
    console.log(`[ChatEngine] Compact mode: ${enabled ? "ON" : "OFF"}`);
  }

  /**
   * Retrieve relevant context from the local knowledge base.
   */
  retrieve(query) {
    const topK = this.compactMode ? Math.min(config.topK, 3) : config.topK;
    return this.store.search(query, topK);
  }

  /**
   * Format retrieved chunks into a context block for the prompt.
   */
  _buildContext(chunks) {
    if (chunks.length === 0) {
      return "No relevant documents found in local knowledge base.";
    }

    return chunks
      .map(
        (c, i) =>
          `--- Document ${i + 1}: ${c.title} [${c.category}] ---\n${c.content}`
      )
      .join("\n\n");
  }

  /**
   * Generate a response for a user query (non-streaming).
   */
  async query(userMessage, history = []) {
    // 1. Retrieve relevant chunks
    const chunks = this.retrieve(userMessage);
    const context = this._buildContext(chunks);

    // 2. Build messages array
    const systemPrompt = this.compactMode ? SYSTEM_PROMPT_COMPACT : SYSTEM_PROMPT;
    const messages = [
      { role: "system", content: systemPrompt },
      {
        role: "system",
        content: `Retrieved context from local knowledge base:\n\n${context}`,
      },
      ...history,
      { role: "user", content: userMessage },
    ];

    // 3. Call the background model over standard HTTP
    const response = await this.chatClient.chat.completions.create({
      model: "phi-3.5-mini",
      messages: messages,
      temperature: 0.1,
      max_tokens: this.compactMode ? 512 : 1024,
    });

    return {
      text: response.choices[0].message.content,
      sources: chunks.map((c) => ({
        title: c.title,
        category: c.category,
        docId: c.doc_id,
        score: Math.round(c.score * 100) / 100,
      })),
    };
  }

  /**
   * Generate a streaming response for a user query.
   * Returns an async iterable of text chunks.
   */
  async *queryStream(userMessage, history = []) {
    // 1. Retrieve relevant chunks
    const chunks = this.retrieve(userMessage);
    const context = this._buildContext(chunks);

    // 2. Build messages array
    const systemPrompt = this.compactMode ? SYSTEM_PROMPT_COMPACT : SYSTEM_PROMPT;
    const messages = [
      { role: "system", content: systemPrompt },
      {
        role: "system",
        content: `Retrieved context from local knowledge base:\n\n${context}`,
      },
      ...history,
      { role: "user", content: userMessage },
    ];

    // Yield sources metadata first
    yield {
      type: "sources",
      data: chunks.map((c) => ({
        title: c.title,
        category: c.category,
        docId: c.doc_id,
        score: Math.round(c.score * 100) / 100,
      })),
    };

    // 3. Stream from the background local model via standard HTTP stream
    const stream = await this.chatClient.chat.completions.create({
      model: "phi-3.5-mini",
      messages: messages,
      temperature: 0.1,
      max_tokens: this.compactMode ? 512 : 1024,
      stream: true,
    });

    for await (const chunk of stream) {
      const content = chunk.choices?.[0]?.delta?.content;
      if (content) {
        yield { type: "text", data: content };
      }
    }
  }

  close() {
    if (this.store) this.store.close();
  }
}