// Auth service (stub)
export async function login(email, password) {
  // Call backend login
  return { token: 'fake-token' };
}

export async function register(email, password) {
  // Call backend register
  return { success: true };
}

export function logout() {
  // Remove token
}
