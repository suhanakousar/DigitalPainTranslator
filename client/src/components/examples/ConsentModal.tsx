import { useState } from 'react';
import ConsentModal from '../ConsentModal';

export default function ConsentModalExample() {
  const [isOpen, setIsOpen] = useState(true);

  return (
    <div className="min-h-96 bg-background relative">
      <ConsentModal
        isOpen={isOpen}
        onAccept={() => {
          console.log('Consent accepted');
          setIsOpen(false);
        }}
        onDecline={() => {
          console.log('Consent declined');
          setIsOpen(false);
        }}
      />
      {!isOpen && (
        <div className="p-8 text-center">
          <p className="text-muted-foreground">Consent modal closed</p>
          <button 
            onClick={() => setIsOpen(true)}
            className="mt-4 px-4 py-2 bg-primary text-primary-foreground rounded-md"
          >
            Show Consent Modal
          </button>
        </div>
      )}
    </div>
  );
}