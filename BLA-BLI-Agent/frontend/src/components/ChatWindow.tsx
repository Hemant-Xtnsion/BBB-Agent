import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from '../types/ChatTypes';
import { sendMessage } from '../api/chatApi';
import MessageBubble from './MessageBubble';
import { RiSendPlaneFill } from 'react-icons/ri';

interface ChatWindowProps {
    onClose: () => void;
}

const ChatWindow: React.FC<ChatWindowProps> = ({ onClose }) => {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [state, setState] = useState<{
        customer_authed?: boolean;
        customer_email?: string;
        order_number?: string;
        intent?: string;
        missing_field?: string;
    }>({});
    const [emailInput, setEmailInput] = useState('');
    const [orderInput, setOrderInput] = useState('');
    const [conversationId] = useState<string>(() => {
        // Generate a unique conversation ID for this session
        return `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    });
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    useEffect(() => {
        // Keep auth form in sync with returned state
        if (state?.customer_email && !emailInput) setEmailInput(state.customer_email);
        if (state?.order_number && !orderInput) setOrderInput(state.order_number);
    }, [state]);

    useEffect(() => {
        // Add welcome message
        const welcomeMessage: ChatMessage = {
            id: '0',
            type: 'bot',
            content:
                "Hi! I'm Gemi, your shopping assistant. How can I help you find the perfect perfume?",
            timestamp: new Date(),
        };
        setMessages([welcomeMessage]);
    }, []);

    const handleSendMessage = async () => {
        if (!inputValue.trim() || isLoading) return;

        const userMessage: ChatMessage = {
            id: Date.now().toString(),
            type: 'user',
            content: inputValue,
            timestamp: new Date(),
        };

        setMessages((prev) => [...prev, userMessage]);
        setInputValue('');
        setIsLoading(true);

        try {
            // Extract email and order number from message if present
            const emailMatch = inputValue.match(/[\w.-]+@[\w.-]+\.\w+/);
            const orderMatch = inputValue.match(/\bBLB-\d{3,}\b/i);
            
            const response = await sendMessage(
                inputValue,
                conversationId,
                state.customer_email || (emailMatch ? emailMatch[0] : undefined),
                state.order_number || (orderMatch ? orderMatch[0].toUpperCase() : undefined)
            );

            const botMessage: ChatMessage = {
                id: (Date.now() + 1).toString(),
                type: 'bot',
                content: response.message,
                products: response.products || [],
                timestamp: new Date(),
            };

            setMessages((prev) => [...prev, botMessage]);
            
            // Update state from response
            if (response.state) {
                setState(response.state);
            }
        } catch (error) {
            console.error('Error sending message:', error);

            const errorMessage: ChatMessage = {
                id: (Date.now() + 1).toString(),
                type: 'bot',
                content:
                    "I'm sorry, I'm having trouble connecting to the server. Please try again in a moment.",
                timestamp: new Date(),
            };

            setMessages((prev) => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    const handleSubmitAuthDetails = async () => {
        const email = emailInput.trim();
        const order = orderInput.trim();
        if (!email || !order || isLoading) return;

        const newState = { ...state, customer_email: email, order_number: order };
        setState(newState);
        setIsLoading(true);

        const userMessage: ChatMessage = {
            id: Date.now().toString(),
            type: 'user',
            content: `Email: ${email}, Order: ${order}`,
            timestamp: new Date(),
        };

        setMessages((prev) => [...prev, userMessage]);

        try {
            const response = await sendMessage(
                `Email: ${email}, Order: ${order}`,
                conversationId,
                email,
                order.toUpperCase()
            );

            const botMessage: ChatMessage = {
                id: (Date.now() + 1).toString(),
                type: 'bot',
                content: response.message,
                products: response.products || [],
                timestamp: new Date(),
            };

            setMessages((prev) => [...prev, botMessage]);
            
            if (response.state) {
                setState(response.state);
            }
        } catch (error) {
            console.error('Error submitting auth details:', error);
            const errorMessage: ChatMessage = {
                id: (Date.now() + 1).toString(),
                type: 'bot',
                content: "I'm sorry, I'm having trouble connecting to the server. Please try again in a moment.",
                timestamp: new Date(),
            };
            setMessages((prev) => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="relative flex flex-col h-full bg-white text-gray-900 rounded-3xl shadow-[0_25px_60px_rgba(149,54,70,0.22)] border border-rose-100 overflow-hidden">
            <div className="relative flex flex-col h-full">
                {/* Header */}
            <header className="bg-gradient-to-r from-[#f25c54] via-[#e9474e] to-[#c81d25] px-5 py-4 flex items-center justify-between border-b border-rose-100 text-white">
                <div className="flex items-center gap-3">
                    <div className="relative">
                        <div className="w-11 h-11 rounded-2xl bg-white/15 border border-white/40 flex items-center justify-center shadow-lg shadow-rose-400/40">
                            <span className="text-sm font-black tracking-[0.15em] text-white">BBB</span>
                        </div>
                        <span className="absolute -bottom-0.5 -right-0.5 w-3.5 h-3.5 rounded-full bg-[#44f0a3] border-2 border-[#f25c54]" />
                    </div>
                    <div>
                        <h1 className="text-lg font-semibold text-white leading-tight">Gemi</h1>
                        <p className="text-xs text-white/80 uppercase tracking-[0.2em]">Luxury concierge</p>
                    </div>
                </div>
                <button
                    onClick={onClose}
                    className="w-10 h-10 flex items-center justify-center text-white/80 hover:text-white hover:bg-white/10 rounded-2xl transition-all duration-200 border border-white/10"
                    aria-label="Close chat"
                >
                    <svg
                        className="w-5 h-5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                    >
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M6 18L18 6M6 6l12 12"
                        />
                    </svg>
                </button>
            </header>

            {/* Messages Container */}
            <div className="flex-1 overflow-y-auto px-5 py-5 bg-rose-50">
                <div className="space-y-4">
                    {messages.map((message) => (
                        <MessageBubble key={message.id} message={message} />
                    ))}

                    {/* Loading Indicator */}
                    {isLoading && (
                        <div className="flex justify-start mb-4">
                            <div className="bg-white border border-rose-100 rounded-2xl rounded-tl-sm px-4 py-3 shadow-sm">
                                <div className="flex gap-1.5">
                                    <div className="w-2 h-2 bg-[#f25c54] rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                                    <div className="w-2 h-2 bg-[#d62839] rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                                    <div className="w-2 h-2 bg-[#f25c54] rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Auth Form for Order Tracking */}
                    {(state?.missing_field === "auth" ||
                        (state?.intent === "order_status" && state?.customer_authed === false)) && (
                        <div className="bg-white border border-rose-100 rounded-2xl p-4 shadow-md shadow-rose-100 mb-4">
                            <div className="text-sm font-semibold text-gray-900 mb-3">Verify your order</div>
                            <div className="space-y-3">
                                <input
                                    type="email"
                                    placeholder="Email used for purchase"
                                    value={emailInput}
                                    onChange={(e) => setEmailInput(e.target.value)}
                                    className="w-full px-3 py-2 text-sm bg-white border border-rose-100 text-gray-900 placeholder:text-gray-400 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#f25c54]/50 focus:border-transparent transition-all"
                                />
                                <input
                                    type="text"
                                    placeholder="Order number (e.g., BLB-1001)"
                                    value={orderInput}
                                    onChange={(e) => setOrderInput(e.target.value)}
                                    className="w-full px-3 py-2 text-sm bg-white border border-rose-100 text-gray-900 placeholder:text-gray-400 rounded-lg focus:outline-none focus:ring-2 focus:ring-[#f25c54]/50 focus:border-transparent transition-all"
                                />
                                <button
                                    onClick={handleSubmitAuthDetails}
                                    disabled={isLoading || !emailInput.trim() || !orderInput.trim()}
                                    className="w-full bg-gradient-to-r from-[#f25c54] to-[#d62839] hover:from-[#ff7c73] hover:to-[#f04646] disabled:from-gray-200 disabled:to-gray-200 text-white font-medium py-2.5 px-4 rounded-xl transition-all duration-200 shadow-lg shadow-rose-200 hover:shadow-rose-300 disabled:text-gray-400 disabled:cursor-not-allowed disabled:shadow-none"
                                >
                                    Submit
                                </button>
                            </div>
                            <div className="text-xs text-gray-500 mt-2">We’ll use these only to fetch your order status.</div>
                        </div>
                    )}

                    <div ref={messagesEndRef} />
                </div>
            </div>

            {/* Input Area */}
            <div className="bg-white border-t border-rose-100 px-4 pt-4 pb-2">
                <div className="relative flex-1">
                    <textarea
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Ask me about perfumes..."
                        rows={1}
                        className="w-full px-4 py-3 pr-16 text-sm text-gray-900 placeholder:text-gray-500 bg-white border border-rose-100 rounded-3xl focus:outline-none focus:ring-2 focus:ring-[#f25c54]/40 focus:border-transparent transition-all resize-none shadow-inner shadow-rose-100"
                        style={{ minHeight: '52px', maxHeight: '120px' }}
                    />
                    <button
                        onClick={handleSendMessage}
                        disabled={!inputValue.trim() || isLoading}
                        className="absolute bottom-2.5 right-2.5 group disabled:cursor-not-allowed"
                        aria-label="Send message"
                    >
                        <span className="absolute inset-0 blur-md rounded-full bg-gradient-to-r from-[#f25c54]/60 to-[#d62839]/60 opacity-0 group-hover:opacity-80 transition-opacity duration-150 pointer-events-none" />
                        <span className="relative flex items-center justify-center w-10 h-10 rounded-full bg-gradient-to-r from-[#f25c54] via-[#e63b44] to-[#c81d25] text-white shadow-lg shadow-rose-300 border border-rose-100 group-hover:scale-[1.05] transition-transform duration-150 disabled:from-gray-200 disabled:via-gray-200 disabled:to-gray-200 disabled:text-gray-400 disabled:shadow-none">
                            <RiSendPlaneFill className="w-4 h-4" />
                        </span>
                    </button>
                </div>
                {/* Powered by Xtnsion.AI */}
                <div className="flex justify-center mt-1">
                    <p className="text-[9px] text-gray-400 tracking-wide">
                        Powered by <span className="font-semibold text-gray-500">Xtnsion.AI</span>
                    </p>
                </div>
            </div>
            </div>
        </div>
    );
};

export default ChatWindow;
