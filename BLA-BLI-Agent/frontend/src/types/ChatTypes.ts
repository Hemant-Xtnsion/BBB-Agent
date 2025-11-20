export interface Product {
    title: string;
    price: string;
    thumbnail: string;
    url: string;
    description?: string;
    tags?: string[];
    category?: string;
}

export interface ChatMessage {
    id: string;
    type: 'user' | 'bot';
    content: string;
    products?: Product[];
    timestamp: Date;
}

export interface ChatResponse {
    type: 'text' | 'products';
    message: string;
    products?: Product[];
    state?: {
        customer_authed?: boolean;
        customer_email?: string;
        order_number?: string;
        intent?: string;
        missing_field?: string;
    };
}

export interface ChatRequest {
    message: string;
    conversation_id?: string;
    email?: string;
    order_number?: string;
}
