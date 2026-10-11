// See https://svelte.dev/docs/kit/types#app
declare global {
  namespace App {
    interface Platform {
      server: {
        upgrade: (req: Request, data?: unknown) => boolean;
        request: Request;
      };
      request: Request;
    }
  }
}

export {};
