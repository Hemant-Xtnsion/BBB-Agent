import React from 'react';

interface WebsiteBackgroundProps {
    screenshotUrl?: string;
}

const WebsiteBackground: React.FC<WebsiteBackgroundProps> = ({ 
    screenshotUrl = 'https://res.cloudinary.com/dkofokw5o/image/upload/v1763645981/bla-bli-web_q9puk7.png'
}) => {
    return (
        <div className="w-full h-full bg-white overflow-y-auto">
            <div className="relative w-full min-h-full">
                <img
                    src={screenshotUrl}
                    alt="Bla Bli Blu Website"
                    className="w-full h-auto object-contain"
                    style={{
                        display: 'block',
                        minHeight: '100vh',
                        width: '100%',
                        objectFit: 'cover',
                        objectPosition: 'top'
                    }}
                    onError={(e) => {
                        // Fallback to white background if image fails to load
                        const target = e.target as HTMLImageElement;
                        target.style.display = 'none';
                    }}
                />
            </div>
        </div>
    );
};

export default WebsiteBackground;

