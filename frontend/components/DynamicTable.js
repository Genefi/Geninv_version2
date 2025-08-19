import React from 'react';

export default function DynamicTable({ columns, rows }) {
  return (
    <table>
      <thead>
        <tr>
          {columns.map((col, idx) => <th key={idx}>{col}</th>)}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, idx) => (
          <tr key={idx}>
            {columns.map((col, cidx) => <td key={cidx}>{row[col]}</td>)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
