import axios from 'axios';
import { ChatRequest, ChatResponse } from '../types/ChatTypes';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
    timeout: 30000,
});

export const sendMessage = async (
    message: string,
    conversationId?: string,
    email?: string,
    orderNumber?: string
): Promise<ChatResponse> => {
    const request: ChatRequest = {
        message,
        conversation_id: conversationId,
        email,
        order_number: orderNumber,
    };

    const response = await apiClient.post<ChatResponse>('/chat', request);
    return response.data;
};

export const checkHealth = async (): Promise<{ status: string }> => {
    const response = await apiClient.get('/health');
    return response.data;
};

export const clearConversation = async (
    conversationId?: string
): Promise<{ status: string; message: string }> => {
    const request: ChatRequest = {
        message: '', // Not used but required by ChatRequest
        conversation_id: conversationId,
    };

    const response = await apiClient.post<{ status: string; message: string }>('/chat/clear', request);
    return response.data;
};