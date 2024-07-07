import React, { useState, useEffect } from "react";
import axios from 'axios';

const ChatBox = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false); 

  const handleSend = async () => {
    if (input.trim()) {
      const newMessages = [...messages, { sender: "user", text: input }];
      setMessages(newMessages);
      setInput("");
      setLoading(true); 

      try {
        const response = await axios.post("http://127.0.0.1:5001/get_question", {
          question: input
        });

        if (response.data.error) {
          setError(response.data.error);
        } else {
          setMessages((prevMessages) => [
            ...prevMessages,
            { sender: "ai", text: response.data.answer },
          ]);
        }
      } catch (error) {
        setError(error.message);
      } finally {
        setLoading(false); 
      }
    }
  };

  useEffect(() => {
    const timeoutId = setTimeout(() => setError(null), 3000);
    return () => clearTimeout(timeoutId);
  }, [error]);

  return (
    <div className="flex flex-col items-center p-4 bg-slate-900  min-h-screen">
      <h1 className="text-3xl mt-16 font-extrabold">Legal contract advisor</h1>
      <div className="bg-slate-800 w-full max-w-xl rounded-lg shadow-lg h-full p-4 mt-16">
        <div className="h-96 overflow-y-auto">
          {messages.map((msg, index) => (
            <div key={index} className={`chat ${msg.sender === "user" ? "chat-end" : "chat-start"}`}>
              <div className="chat-bubble bg-white text-black">
                {msg.text}
              </div>
            </div>
          ))}
          {error && <p className="text-red-500 mt-2">{error}</p>}
        </div>
        <div className="flex mt-4">
          <input
            type="text"
            className="input input-bordered flex-grow bg-white text-black"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && handleSend()}
          />
          <button
            className="btn btn-primary ml-2 bg-white text-black"
            onClick={handleSend}
            disabled={loading} // Disable button when loading
          >
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatBox;
