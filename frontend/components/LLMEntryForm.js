import React, { useState } from 'react';

export default function LLMEntryForm({ onSubmit, onLLMParse }) {
  const [form, setForm] = useState({ name: '', sku: '', quantity: '', price: '' });
  const [nlInput, setNlInput] = useState('');
  const [llmSuggestion, setLlmSuggestion] = useState(null);

  const handleLLM = async () => {
    const suggestion = await onLLMParse(nlInput);
    setLlmSuggestion(suggestion);
    setForm(suggestion);
  };

  return (
    <div>
      <h3>Add Item (LLM Assisted)</h3>
      <input placeholder="Natural language input" value={nlInput} onChange={e => setNlInput(e.target.value)} />
      <button onClick={handleLLM}>Parse with LLM</button>
      <form onSubmit={e => { e.preventDefault(); onSubmit(form); }}>
        <input placeholder="Item Name" value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} />
        <input placeholder="SKU" value={form.sku} onChange={e => setForm({ ...form, sku: e.target.value })} />
        <input placeholder="Quantity" value={form.quantity} onChange={e => setForm({ ...form, quantity: e.target.value })} />
        <input placeholder="Price" value={form.price} onChange={e => setForm({ ...form, price: e.target.value })} />
        <button type="submit">Add Item</button>
      </form>
      {llmSuggestion && <div>LLM Suggestion: {JSON.stringify(llmSuggestion)}</div>}
    </div>
  );
}
