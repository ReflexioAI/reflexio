import { redirect } from "next/navigation";

export default function Home() {
  redirect("/interactions/get-all-interactions");
}
