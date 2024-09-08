import React, { useState } from 'react';
import './Chatbot.css';

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (input.trim()) {
      setMessages([...messages, { text: input, user: 'user' }]);
      getBotResponse(input);
      setInput('');
    }
  };

  const getBotResponse = (userInput) => {
    // Simulate a bot response
    const botResponse = `Bot says: ${userInput}`;
    setMessages((prevMessages) => [
      ...prevMessages,
      { text: botResponse, user: 'bot' },
    ]);
  };

  return (
    <div className="chatbot">
      <div className="chat-window">
        {messages.map((message, index) => (
          <div
            key={index}
            className={`chat-message ${message.user}`}
          >
            {message.text}
          </div>
        ))}
      </div>
      <div className="chat-input">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Type a message..."
        />
        <button onClick={handleSend}>Send</button>
      </div>
    </div>
  );
};

export default Chatbot;
