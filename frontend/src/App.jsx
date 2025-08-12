// frontend/src/App.jsx
import React, { useState, useRef, useEffect } from "react";

export default function App() {
  const [messages, setMessages] = useState([
    { from: "bot", text: "Hello! I'm a melting chatbot. Ask me anything." },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage() {
    const content = input.trim();
    if (!content) return;
    // push user message
    setMessages((prev) => [...prev, { from: "user", text: content }]);
    setInput("");
    setLoading(true);

    // push an empty bot message that we'll fill while streaming
    setMessages((prev) => [...prev, { from: "bot", text: "" }]);

    try {
      const res = await fetch("/api/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: content }),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let done = false;

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        if (value) {
          const chunk = decoder.decode(value, { stream: true });
          // append chunk to last bot message
          setMessages((prev) => {
            const copy = [...prev];
            // find last message (should be the empty bot message)
            const lastIdx = copy.length - 1;
            if (lastIdx >= 0 && copy[lastIdx].from === "bot") {
              copy[lastIdx] = { ...copy[lastIdx], text: copy[lastIdx].text + chunk };
            } else {
              // fallback: push a new bot message
              copy.push({ from: "bot", text: chunk });
            }
            return copy;
          });
        }
      }
    } catch (err) {
      // append an error message from the bot
      setMessages((prev) => [...prev, { from: "bot", text: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-6">
      <div className="w-full max-w-xl bg-white rounded-2xl shadow-lg p-6 grid grid-rows-[auto_1fr_auto]" style={{ height: "80vh" }}>
        <header className="mb-4 flex items-center gap-3">
          <div className="rounded-full bg-indigo-600 text-white w-10 h-10 flex items-center justify-center font-bold">CB</div>
          <div>
            <h1 className="text-lg font-semibold">Melt Bot</h1>
            <p className="text-sm text-gray-500">dna-utah.org</p>
          </div>
        </header>

        <main className="overflow-auto p-2" style={{ borderTop: "1px solid #f3f4f6", borderBottom: "1px solid #f3f4f6" }}>
          <div className="flex flex-col gap-3 p-2">
            {messages.map((m, i) => (
              <div key={i} className={`flex ${m.from === "user" ? "justify-end" : "justify-start"}`}>
                <div className={`${m.from === "user" ? "bg-indigo-600 text-white" : "bg-gray-100 text-gray-900"} max-w-[80%] p-3 rounded-lg`}>
                  <div className="whitespace-pre-wrap">{m.text}</div>
                </div>
              </div>
            ))}
            <div ref={endRef} />
          </div>
        </main>

        <footer className="mt-4">
          <div className="flex gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Type your message (Enter to send)"
              className="resize-none p-3 rounded-lg border border-gray-200 flex-1 h-14 focus:outline-none"
            />
            <button onClick={sendMessage} disabled={loading} className="bg-indigo-600 text-white px-4 py-2 rounded-lg disabled:opacity-60">
              {loading ? "Generating..." : "Send"}
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}
