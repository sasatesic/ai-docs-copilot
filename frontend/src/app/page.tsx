"use client";

import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, Bot, User, Loader2, Paperclip, FileText } from 'lucide-react';

const API_URL = "/api";

interface Source {
  score: number;
  text: string;
  source_file?: string;
  source_id?: string;
}

interface Message {
  role: 'assistant' | 'user';
  content: string;
  sources?: Source[];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', content: 'Hello! I am your AI Docs Copilot. Ask me anything about your documents.' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Create a placeholder for the AI response
    const aiMessagePlaceholder: Message = { role: 'assistant', content: '', sources: [] };
    setMessages(prev => [...prev, aiMessagePlaceholder]);

    try {
      const response = await fetch(`${API_URL}/chat_stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage.content })
      });

      if (!response.ok) throw new Error('Network response was not ok');
      if (!response.body) return;

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        
        buffer = lines.pop() || ''; 

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const parsed = JSON.parse(line);
            
            if (parsed.type === 'sources') {
              setMessages(prev => {
                const newArr = [...prev];
                const lastIndex = newArr.length - 1;
                // FIX: Create a shallow copy of the last message object
                newArr[lastIndex] = { ...newArr[lastIndex], sources: parsed.data };
                return newArr;
              });
            } else if (parsed.type === 'content') {
              setMessages(prev => {
                const newArr = [...prev];
                const lastIndex = newArr.length - 1;
                // FIX: Create a shallow copy of the last message object
                // This prevents "doubling" in Strict Mode
                newArr[lastIndex] = { 
                  ...newArr[lastIndex], 
                  content: newArr[lastIndex].content + parsed.data 
                };
                return newArr;
              });
            }
          } catch (e) {
            console.error("Error parsing stream:", e);
          }
        }
      }
    } catch (error) {
      setMessages(prev => {
        const newArr = [...prev];
        const lastIndex = newArr.length - 1;
        newArr[lastIndex] = {
          ...newArr[lastIndex],
          content: newArr[lastIndex].content + "\n\n**Error: Could not reach the server.**"
        };
        return newArr;
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    
    setSelectedFile(file.name);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const res = await fetch(`${API_URL}/documents`, {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        alert(`Successfully uploaded ${file.name}`);
      } else {
        alert('Upload failed. Check backend logs.');
      }
    } catch (err) {
      alert('Error uploading file');
    }
    setSelectedFile(null);
  };

  return (
    <main className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 p-4 flex items-center justify-center shadow-sm sticky top-0 z-10">
        <div className="flex items-center gap-2">
          <div className="bg-blue-600 p-1.5 rounded-lg">
            <Bot className="w-5 h-5 text-white" />
          </div>
          <h1 className="font-bold text-lg text-gray-800">AI Docs Copilot</h1>
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-6 max-w-4xl mx-auto w-full">
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
            
            <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 shadow-sm
              ${msg.role === 'user' ? 'bg-blue-600' : 'bg-white border border-gray-200'}`}>
              {msg.role === 'user' ? <User className="w-4 h-4 text-white" /> : <Bot className="w-4 h-4 text-blue-600" />}
            </div>

            <div className={`max-w-[85%] space-y-2`}>
              <div className={`p-4 rounded-2xl shadow-sm text-sm leading-relaxed
                ${msg.role === 'user' 
                  ? 'bg-blue-600 text-white rounded-tr-none' 
                  : 'bg-white border border-gray-200 text-gray-800 rounded-tl-none'}`}>
                <div className="prose prose-sm max-w-none">
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                </div>
              </div>

              {/* Sources */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {msg.sources.slice(0, 3).map((source, sIdx) => (
                    <div key={sIdx} className="text-xs bg-white text-gray-600 px-2 py-1 rounded border border-gray-200 flex items-center gap-1 shadow-sm">
                      <FileText className="w-3 h-3 text-blue-500" />
                      <span className="font-medium truncate max-w-[150px]">{source.source_id || source.source_file}</span>
                      <span className="text-gray-400 pl-1 border-l border-gray-200 ml-1">{Math.round(source.score * 100)}%</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="bg-white border-t border-gray-200 p-4">
        <div className="max-w-4xl mx-auto flex gap-3 items-center">
          <label className="cursor-pointer p-2 text-gray-400 hover:text-blue-600 hover:bg-blue-50 rounded-full transition-all" title="Upload Document">
            <input type="file" className="hidden" onChange={handleFileChange} accept=".pdf,.txt,.md,.docx,.pptx,.xlsx" />
            <Paperclip className="w-5 h-5" />
          </label>
          
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask a question about your documents..."
            className="flex-1 p-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all text-gray-800 placeholder-gray-400"
            disabled={isLoading}
          />
          
          <button 
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="bg-blue-600 text-white p-3 rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-sm"
          >
            {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
          </button>
        </div>
        {selectedFile && <div className="max-w-4xl mx-auto mt-2 text-xs text-green-600 font-medium flex items-center gap-1">✓ Uploading: {selectedFile}...</div>}
      </div>
    </main>
  );
}