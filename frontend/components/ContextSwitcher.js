import React from 'react';

export default function ContextSwitcher({ tables, current, onSwitch }) {
  return (
    <div>
      <label>Active Table: </label>
      <select value={current} onChange={e => onSwitch(e.target.value)}>
        {tables.map((t, idx) => <option key={idx} value={t}>{t}</option>)}
      </select>
    </div>
  );
}
