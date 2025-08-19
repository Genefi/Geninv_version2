import React from 'react';

export default function InventoryTable({ items }) {
  return (
    <table>
      <thead>
        <tr>
          <th>Item Name</th>
          <th>SKU</th>
          <th>Quantity</th>
          <th>Price</th>
        </tr>
      </thead>
      <tbody>
        {items && items.map((item, idx) => (
          <tr key={idx}>
            <td>{item.name}</td>
            <td>{item.sku}</td>
            <td>{item.quantity}</td>
            <td>{item.price}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
