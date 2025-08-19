// API service for backend communication
export async function fetchTables() {
  // Fetch dynamic tables from backend
  return [];
}

export async function addItem(table, item) {
  // Add item to table
  return { success: true };
}

export async function parseWithLLM(nlInput) {
  // Call LLM parse endpoint
  return { name: 'Example', sku: 'EX-001', quantity: 10, price: 20 };
}
