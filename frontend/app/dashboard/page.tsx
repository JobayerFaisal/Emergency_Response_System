"use client";

import { useEffect, useState } from "react";
import EnvironmentalPanel from "./components/EnvironmentalPanel";

type DispatchOrder = {
  team_id: string;
  team_name: string;
  target_lat: number;
  target_lon: number;
  requester_name: string;
  requester_phone: string;
  details: string;
};

// For now keep it simple: backend is exposed on localhost:8000
const API_BASE = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/ws/dispatches";

export default function DashboardPage() {
  const [dispatches, setDispatches] = useState<DispatchOrder[]>([]);
  const [wsStatus, setWsStatus] = useState("Connecting...");
  const [formStatus, setFormStatus] = useState<string | null>(null);

  // Form state
  const [name, setName] = useState("Fahim");
  const [phone, setPhone] = useState("01700000000");
  const [lat, setLat] = useState(23.8103);
  const [lon, setLon] = useState(90.4125);
  const [details, setDetails] = useState("Water rising fast, need boat.");

  // WebSocket connection
  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        setWsStatus("Connected. Waiting for dispatches...");
      };

      ws.onmessage = (event) => {
        setWsStatus("Receiving dispatches in real time.");
        try {
          const data = JSON.parse(event.data) as DispatchOrder;
          setDispatches((prev) => [data, ...prev]);
        } catch (err) {
          console.error("Failed to parse dispatch:", err, event.data);
        }
      };

      ws.onclose = () => {
        setWsStatus("Disconnected. Reconnecting in 3s...");
        reconnectTimeout = setTimeout(connect, 3000);
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
      };
    };

    connect();

    return () => {
      if (ws) ws.close();
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormStatus("Sending request...");

    try {
      const res = await fetch(`${API_BASE}/api/v1/rescue-requests/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          phone,
          lat,
          lon,
          details,
        }),
      });

      if (!res.ok) {
        throw new Error(`Request failed with status ${res.status}`);
      }

      const data = await res.json();
      console.log("Rescue request response:", data);
      setFormStatus("Rescue request sent successfully!");
    } catch (error) {
      console.error(error);
      setFormStatus("Failed to send rescue request.");
    }
  };

  return (
    <main className="min-h-screen bg-slate-100 p-6 flex flex-col md:flex-row gap-8">
      {/* Left column: form */}
      <section className="w-full md:w-1/3 bg-white rounded-2xl shadow-md p-6">
        <h1 className="text-2xl font-bold mb-4">Rescue Request</h1>
        <form onSubmit={handleSubmit} className="space-y-3">
          <div>
            <label className="block text-sm font-medium mb-1">Name</label>
            <input
              className="w-full border rounded-md px-3 py-2 text-sm"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Phone</label>
            <input
              className="w-full border rounded-md px-3 py-2 text-sm"
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              required
            />
          </div>

          <div className="flex gap-3">
            <div className="flex-1">
              <label className="block text-sm font-medium mb-1">
                Latitude
              </label>
              <input
                type="number"
                step="0.0001"
                className="w-full border rounded-md px-3 py-2 text-sm"
                value={lat}
                onChange={(e) => setLat(parseFloat(e.target.value))}
                required
              />
            </div>
            <div className="flex-1">
              <label className="block text-sm font-medium mb-1">
                Longitude
              </label>
              <input
                type="number"
                step="0.0001"
                className="w-full border rounded-md px-3 py-2 text-sm"
                value={lon}
                onChange={(e) => setLon(parseFloat(e.target.value))}
                required
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium mb-1">Details</label>
            <textarea
              rows={3}
              className="w-full border rounded-md px-3 py-2 text-sm"
              value={details}
              onChange={(e) => setDetails(e.target.value)}
            />
          </div>

          <button
            type="submit"
            className="w-full bg-blue-600 text-white rounded-md py-2 text-sm font-semibold hover:bg-blue-700 transition"
          >
            Request Rescue
          </button>

          {formStatus && (
            <p className="text-xs text-slate-600 mt-1">{formStatus}</p>
          )}
        </form>
      </section>

      {/* Right column: live dispatches + environmental panel */}
      <section className="w-full md:flex-1 flex flex-col gap-6">
        {/* Live dispatches card */}
        <div className="bg-white rounded-2xl shadow-md p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold">Live Dispatches</h2>
            <span className="text-xs text-slate-600">{wsStatus}</span>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full text-sm border">
              <thead className="bg-slate-50">
                <tr>
                  <th className="border px-2 py-1 text-left">Team</th>
                  <th className="border px-2 py-1 text-left">Requester</th>
                  <th className="border px-2 py-1 text-left">Phone</th>
                  <th className="border px-2 py-1 text-left">Location</th>
                  <th className="border px-2 py-1 text-left">Details</th>
                </tr>
              </thead>
              <tbody>
                {dispatches.length === 0 && (
                  <tr>
                    <td
                      colSpan={5}
                      className="border px-2 py-4 text-center text-slate-500"
                    >
                      No dispatches yet. Submit a rescue request to see it live.
                    </td>
                  </tr>
                )}
                {dispatches.map((d, index) => (
                  <tr key={index} className="hover:bg-slate-50">
                    <td className="border px-2 py-1">
                      {d.team_id} – {d.team_name}
                    </td>
                    <td className="border px-2 py-1">{d.requester_name}</td>
                    <td className="border px-2 py-1">{d.requester_phone}</td>
                    <td className="border px-2 py-1">
                      {d.target_lat.toFixed(4)}, {d.target_lon.toFixed(4)}
                    </td>
                    <td className="border px-2 py-1">{d.details}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Environmental predictions card */}
        <div className="bg-white rounded-2xl shadow-md p-6">
          <EnvironmentalPanel />
        </div>
      </section>
    </main>
  );
}
