import { Bench } from '@/components/Bench';
import { Masthead } from '@/components/Masthead';

export default function App() {
  return (
    /*
     * The Masthead sits outside <main> so it is a real `banner` landmark — a <header> nested inside
     * <main> is not one, and a screen-reader user skipping to the page's header would have landed
     * nowhere. `min-h-full` moves out to this wrapper along with it: left on <main>, the masthead's
     * height would be added to a box already asking for the full viewport and the page would
     * scroll by exactly that much on every screen.
     */
    <div className="min-h-full bg-[var(--color-bg)]">
      <Masthead />
      <main>
        <Bench />
      </main>
    </div>
  );
}
