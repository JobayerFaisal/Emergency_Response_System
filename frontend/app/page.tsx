// frontend/app/page.tsx
import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen flex items-center justify-center bg-slate-900 text-white">
      <div className="text-center space-y-4">
        <h1 className="text-3xl font-bold">
          Emergency Response System Dashboard
        </h1>
        <p>Go to the live rescue dashboard:</p>
        <Link
          href="/dashboard"
          className="inline-block bg-blue-600 px-4 py-2 rounded-md text-sm font-semibold hover:bg-blue-700"
        >
          Open Dashboard
        </Link>
      </div>
    </main>
  );
}
