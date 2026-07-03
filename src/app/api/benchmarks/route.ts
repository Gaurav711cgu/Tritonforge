import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET() {
  const possiblePaths = [
    path.join(process.cwd(), "..", "tritonforge", "tritonforge", "benchmarks", "results_T4.json"),
    path.join(process.cwd(), "..", "tritonforge", "benchmarks", "results_T4.json"),
    path.join(process.cwd(), "benchmarks", "results_T4.json"),
  ];

  for (const p of possiblePaths) {
    if (fs.existsSync(p)) {
      try {
        const raw = fs.readFileSync(p, "utf-8");
        return NextResponse.json(JSON.parse(raw));
      } catch (err) {
        console.error("Error reading benchmarks file at", p, err);
      }
    }
  }

  return NextResponse.json({ error: "Benchmarks file results_T4.json not found" }, { status: 404 });
}
