// Bun loads `.sql` files as text when imported with `{ type: "text" }`.
declare module "*.sql" {
  const content: string;
  export default content;
}
