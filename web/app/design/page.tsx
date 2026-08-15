import { Masthead } from '../Masthead';
import { Designer } from '../../components/Designer';

export const metadata = { title: 'Design a level · Snake Agent' };

export default function DesignPage() {
  return (
    <main className="shell">
      <Masthead />
      <p className="eyebrow">Level designer</p>
      <h1>Build something it cannot finish.</h1>
      <p className="lede">
        Drag to place walls, then set the agent loose on your board. It has never seen this layout —
        it was trained on randomly generated obstacles precisely so it would have to learn walls as
        an idea rather than memorise four maps. The amber cells are where the snake starts, so they
        stay clear.
      </p>
      <Designer />
    </main>
  );
}
