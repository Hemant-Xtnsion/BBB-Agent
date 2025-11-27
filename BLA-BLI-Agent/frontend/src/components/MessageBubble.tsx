import React, { CSSProperties } from 'react';
import { ChatMessage } from '../types/ChatTypes';
import ProductCard from './ProductCard';
import { RiUserFill } from 'react-icons/ri';

interface MessageBubbleProps {
    message: ChatMessage;
    onButtonClick?: (action: string, label: string) => void;
    isLatest?: boolean;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ message, onButtonClick, isLatest = true }) => {
    const isUser = message.type === 'user';

    return (
        <div
            className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4 animate-slide-up`}
        >
            <div className={`flex items-start gap-3 max-w-[85%] ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
                {/* Avatar */}
                <div
                    className={`flex-shrink-0 ${isUser
                        ? 'w-9 h-9 rounded-full bg-gradient-to-br from-[#f46a6f] to-[#d62839] text-white border border-[#fbd0d4]'
                        : 'w-9 h-9 rounded-2xl bg-white text-[#c01f2f] border border-[#f3c0c6]'
                        } flex items-center justify-center shadow-lg shadow-rose-100/70`}
                >
                    {isUser ? (
                        <RiUserFill className="w-4 h-4" />
                    ) : (
                        <span className="text-[9px] font-black tracking-[0.25em]">BBB</span>
                    )}
                </div>

                {/* Message Content */}
                <div className="flex flex-col gap-2 flex-1 text-gray-900">
                    {/* Text Bubble */}
                    <div
                        className={`px-4 py-3 rounded-3xl shadow-md ${isUser
                            ? 'bg-gradient-to-br from-[#f25c54] via-[#d62839] to-[#8b152d] text-white rounded-tr-sm shadow-rose-200/70'
                            : 'bg-white text-gray-900 border border-rose-100 rounded-tl-sm shadow-rose-100/60'
                            }`}
                    >
                        <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                    </div>

                    {/* Product Cards - Compact grid layout, no scrollbar */}
                    {message.products && message.products.length > 0 && (
                        <div className="mt-3">
                            <div className="grid grid-cols-2 gap-3">
                                {message.products.map((product, index) => (
                                    <ProductCard
                                        key={index}
                                        product={product}
                                        compact={true}
                                        buttons={message.buttons}
                                        onButtonClick={onButtonClick}
                                        isLatest={isLatest}
                                    />
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Button Suggestions - Filter out View Product and bolder/safer buttons when products are shown */}
                    {!isUser && message.buttons && message.buttons.length > 0 && (
                        <div className="mt-1 flex flex-wrap gap-1.5">
                            {message.buttons
                                .filter(button => {
                                    // Hide View Product button if products are shown
                                    if (message.products && message.products.length > 0) {
                                        if (button.action.startsWith('open_url:') || button.label.toLowerCase().includes('view product')) {
                                            return false;
                                        }
                                        // Hide bolder/safer buttons as they're shown in the card
                                        if (button.action === 'show_bolder' || button.action === 'show_safer') {
                                            return false;
                                        }
                                    }
                                    return true;
                                })
                                .map((button, index) => {
                                    type SuggestionStyle = CSSProperties & { '--suggestion-delay'?: string };
                                    const delayMs = index * 300;
                                    const buttonStyle: SuggestionStyle = {
                                        '--suggestion-delay': `${delayMs}ms`,
                                    };
                                    return (
                                        <button
                                            key={index}
                                            onClick={() => isLatest && onButtonClick?.(button.action, button.label)}
                                            disabled={!isLatest}
                                            style={buttonStyle}
                                            className={`
                                            suggestion-chip relative px-4 py-1.5 rounded-full text-xs font-semibold tracking-wide
                                            transition-all duration-200 whitespace-nowrap focus:outline-none focus-visible:ring-2 focus-visible:ring-rose-200/70
                                            ${isLatest
                                                    ? 'bg-white text-rose-600 border border-rose-200 hover:bg-rose-50 hover:border-rose-300 hover:text-rose-700 hover:-translate-y-0.5 hover:shadow-lg shadow-rose-100 cursor-pointer'
                                                    : 'is-disabled bg-gray-50 text-gray-400 border border-gray-200 cursor-not-allowed opacity-60'
                                                }
                                        `}
                                        >
                                            <span className="relative z-[1]">{button.label}</span>
                                        </button>
                                    );
                                })}
                        </div>
                    )}

                    {/* Timestamp */}
                    <span className={`text-xs text-gray-500 ${isUser ? 'text-right' : 'text-left'}`}>
                        {message.timestamp.toLocaleTimeString([], {
                            hour: '2-digit',
                            minute: '2-digit',
                        })}
                    </span>
                </div>
            </div>
        </div>
    );
};

export default MessageBubble;
