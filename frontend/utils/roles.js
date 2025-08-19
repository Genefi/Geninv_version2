// User roles
export const ADMIN = 'admin';
export const USER = 'user';

export function isAdmin(role) {
  return role === ADMIN;
}
