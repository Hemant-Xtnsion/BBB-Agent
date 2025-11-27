import React from 'react';
import { Product, ButtonSuggestion } from '../types/ChatTypes';

interface ProductCardProps {
    product: Product;
    compact?: boolean;
    buttons?: ButtonSuggestion[];
    onButtonClick?: (action: string, label: string) => void;
    isLatest?: boolean;
}

const ProductCard: React.FC<ProductCardProps> = ({ product, compact = false, buttons = [], onButtonClick, isLatest = true }) => {
    const handleViewProduct = () => {
        window.open(product.url, '_blank', 'noopener,noreferrer');
    };

    const handleAddToCart = () => {
        // Dummy function - would integrate with actual cart in bolder
        alert(`Added "${product.title}" to cart!`);
    };

    // Filter bolder/safer buttons
    const bolderButton = buttons.find(b => b.action === 'show_bolder');
    const saferButton = buttons.find(b => b.action === 'show_safer');

    if (compact) {
        // Compact card for chat window - professional design with visible thumbnails
        return (
            <div className="bg-white text-gray-900 rounded-xl shadow-md shadow-rose-100 hover:shadow-rose-200 transition-all duration-200 overflow-hidden border border-rose-100 hover:border-rose-200 group flex flex-col">
                {/* Product Image - Larger and more prominent */}
                <div className="relative w-full h-32 bg-gradient-to-br from-rose-50 via-white to-rose-100 overflow-hidden flex items-center justify-center">
                    {product.thumbnail ? (
                        <img
                            src={product.thumbnail}
                            alt={product.title}
                            className="w-full h-full object-contain group-hover:scale-105 transition-transform duration-300"
                            onError={(e) => {
                                const target = e.target as HTMLImageElement;
                                target.src = 'https://via.placeholder.com/200x200?text=Product';
                            }}
                        />
                    ) : (
                        <div className="w-full h-full flex items-center justify-center bg-rose-100">
                            <svg
                                className="w-12 h-12 text-rose-300"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                                />
                            </svg>
                        </div>
                    )}
                    
                    {/* Price Badge - Top Right */}
                    {product.price && (
                        <div className="absolute top-2 right-2 bg-white/95 backdrop-blur-sm px-1.5 py-0.5 rounded-md shadow-md border border-rose-100">
                            <span className="text-[#c01f2f] font-bold text-[9px] leading-tight flex items-center gap-0.5">
                                <span>₹</span>
                                <span>{product.price.replace(/[₹,\s]/g, '')}</span>
                            </span>
                        </div>
                    )}
                </div>

                {/* Product Info */}
                <div className="p-3 flex flex-col flex-1">
                    <h3 className="font-semibold text-gray-900 text-xs mb-2 line-clamp-2 group-hover:text-[#c01f2f] transition-colors leading-snug">
                        {product.title}
                    </h3>
                    
                    {/* Action Buttons */}
                    <div className="flex flex-col gap-1.5 mt-auto">
                        <button
                            onClick={isLatest ? handleViewProduct : undefined}
                            disabled={!isLatest}
                            className={`w-full font-semibold py-1.5 px-2 rounded-md text-[10px] leading-tight shadow-sm transition-colors ${
                                isLatest 
                                    ? 'bg-white text-[#c01f2f] border border-rose-100 hover:bg-rose-50 cursor-pointer' 
                                    : 'bg-gray-50 text-gray-400 border border-gray-200 cursor-not-allowed opacity-60'
                            }`}
                        >
                            View Product
                        </button>
                        <button
                            onClick={isLatest ? handleAddToCart : undefined}
                            disabled={!isLatest}
                            className={`w-full font-medium py-1.5 px-2 rounded-md text-[10px] leading-tight shadow-sm transition-colors ${
                                isLatest 
                                    ? 'bg-gradient-to-r from-[#f25c54] to-[#d62839] text-white hover:from-[#ff7c73] hover:to-[#f04646] cursor-pointer' 
                                    : 'bg-gray-200 text-gray-400 cursor-not-allowed opacity-60'
                            }`}
                        >
                            Add to Cart
                        </button>
                        
                        {/* Bolder/Safer Option Buttons */}
                        {(bolderButton || saferButton) && (
                            <div className="flex gap-1.5 mt-1">
                                {saferButton && (
                                    <button
                                        onClick={isLatest ? () => onButtonClick?.(saferButton.action, saferButton.label) : undefined}
                                        disabled={!isLatest}
                                        className={`flex-1 font-semibold py-1.5 px-2 rounded-md text-[10px] leading-tight shadow-sm transition-colors ${
                                            isLatest 
                                                ? 'bg-white text-[#c01f2f] border border-rose-200 hover:bg-rose-50 cursor-pointer' 
                                                : 'bg-gray-50 text-gray-400 border border-gray-200 cursor-not-allowed opacity-60'
                                        }`}
                                    >
                                        {saferButton.label.replace(' Option', '')}
                                    </button>
                                )}
                                {bolderButton && (
                                    <button
                                        onClick={isLatest ? () => onButtonClick?.(bolderButton.action, bolderButton.label) : undefined}
                                        disabled={!isLatest}
                                        className={`flex-1 font-semibold py-1.5 px-2 rounded-md text-[10px] leading-tight shadow-sm transition-colors ${
                                            isLatest 
                                                ? 'bg-gradient-to-r from-[#f25c54] to-[#d62839] text-white hover:from-[#ff7c73] hover:to-[#f04646] cursor-pointer' 
                                                : 'bg-gray-200 text-gray-400 cursor-not-allowed opacity-60'
                                        }`}
                                    >
                                        {bolderButton.label.replace(' Option', '')}
                                    </button>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        );
    }

    // Full card for larger displays
    return (
        <div className="bg-white rounded-xl shadow-md hover:shadow-xl transition-all duration-300 overflow-hidden border border-gray-100 hover:border-primary-300 group">
            {/* Product Image */}
            <div className="relative aspect-square w-full bg-gray-100 overflow-hidden">
                {product.thumbnail ? (
                    <img
                        src={product.thumbnail}
                        alt={product.title}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                        onError={(e) => {
                            const target = e.target as HTMLImageElement;
                            target.src = 'https://via.placeholder.com/400x300?text=Product+Image';
                        }}
                    />
                ) : (
                    <div className="w-full h-full flex items-center justify-center bg-gradient-to-br from-primary-100 to-primary-200">
                        <svg
                            className="w-16 h-16 text-primary-400"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                            />
                        </svg>
                    </div>
                )}

                {/* Price Badge */}
                {product.price && (
                    <div className="absolute top-3 right-3 bg-white/95 backdrop-blur-sm px-3 py-1.5 rounded-full shadow-lg">
                        <span className="text-primary-700 font-semibold text-sm">
                            {product.price}
                        </span>
                    </div>
                )}
            </div>

            {/* Product Info */}
            <div className="p-4">
                <h3 className="font-semibold text-gray-900 text-lg mb-2 line-clamp-2 group-hover:text-primary-600 transition-colors">
                    {product.title}
                </h3>

                {product.description && (
                    <p className="text-gray-600 text-sm mb-3 line-clamp-2">
                        {product.description}
                    </p>
                )}

                {/* Tags */}
                {product.tags && product.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1.5 mb-3">
                        {product.tags.slice(0, 3).map((tag, index) => (
                            <span
                                key={index}
                                className="px-2 py-0.5 bg-primary-50 text-primary-700 text-xs rounded-full font-medium"
                            >
                                {tag}
                            </span>
                        ))}
                    </div>
                )}

                {/* Action Buttons */}
                <div className="flex gap-2 mt-4">
                    <button
                        onClick={handleViewProduct}
                        className="flex-1 bg-primary-600 hover:bg-primary-700 text-white font-medium py-2.5 px-4 rounded-lg transition-colors duration-200 shadow-sm hover:shadow-md"
                    >
                        View Product
                    </button>
                    <button
                        onClick={handleAddToCart}
                        className="flex-1 bg-gray-100 hover:bg-gray-200 text-gray-800 font-medium py-2.5 px-4 rounded-lg transition-colors duration-200 border border-gray-200"
                    >
                        Add to Cart
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ProductCard;
