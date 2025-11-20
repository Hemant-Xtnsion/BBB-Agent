import React, { useState } from 'react';
import ChatWindow from './components/ChatWindow';
import WebsiteBackground from './components/WebsiteBackground';

function App() {
    const [isChatOpen, setIsChatOpen] = useState(false);

    return (
        <div className="relative w-full h-screen overflow-hidden">
            {/* Website Background */}
            <WebsiteBackground />

            {/* Floating Chat Button */}
            {!isChatOpen && (
                <button
                    onClick={() => setIsChatOpen(true)}
                    className="fixed bottom-4 right-4 md:bottom-6 md:right-6 w-16 h-16 md:w-20 md:h-20 bg-gradient-to-r from-[#f25c54] via-[#e9474e] to-[#c81d25] rounded-2xl shadow-2xl flex items-center justify-center transition-all duration-300 hover:scale-110 z-50 animate-pulse border border-white/20"
                    aria-label="Open chat"
                >
                    <div className="relative">
                        <div className="w-12 h-12 md:w-14 md:h-14 rounded-xl bg-white/15 border border-white/40 flex items-center justify-center shadow-lg">
                            <span className="text-base md:text-lg font-black tracking-[0.15em] text-white">BBB</span>
                        </div>
                        <span className="absolute -bottom-0.5 -right-0.5 w-3.5 h-3.5 md:w-4 md:h-4 rounded-full bg-[#44f0a3] border-2 border-[#f25c54]" />
                    </div>
                </button>
            )}

            {/* Floating Chat Window */}
            {isChatOpen && (
                <div className="fixed bottom-4 right-4 md:bottom-6 md:right-6 w-[calc(100vw-2rem)] md:w-96 h-[600px] max-h-[calc(100vh-2rem)] z-50 shadow-2xl rounded-t-2xl overflow-hidden animate-slide-up">
                    <ChatWindow onClose={() => setIsChatOpen(false)} />
                </div>
            )}
        </div>
    );
}

export default App;
