export interface Product {
    title: string;
    price: string;
    thumbnail: string;
    url: string;
    description?: string;
    tags?: string[];
    category?: string;
}

export interface ButtonSuggestion {
    label: string;
    action: string;
    type: 'primary' | 'secondary' | 'success' | 'danger';
}

export interface ChatMessage {
    id: string;
    type: 'user' | 'bot';
    content: string;
    products?: Product[];
    buttons?: ButtonSuggestion[];
    timestamp: Date;
}

export interface ChatResponse {
    type: 'text' | 'products';
    message: string;
    products?: Product[];
    buttons?: ButtonSuggestion[];
    state?: {
        customer_authed?: boolean;
        customer_email?: string;
        order_number?: string;
        intent?: string;
        missing_field?: string;
        button_suggestions?: ButtonSuggestion[];
        awaiting_choice?: string;
    };
}

export interface ChatRequest {
    message: string;
    conversation_id?: string;
    email?: string;
    order_number?: string;
}
