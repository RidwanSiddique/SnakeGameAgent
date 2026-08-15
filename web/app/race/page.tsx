import { Masthead } from '../Masthead';
import { Race } from '../../components/Race';

export const metadata = { title: 'Race the agent · Snake Agent' };

export default function RacePage() {
  return (
    <main className="shell">
      <Masthead />
      <p className="eyebrow">Head to head</p>
      <h1>Same board. Same food.</h1>
      <p className="lede">
        Both snakes start from one seed, so the food appears in the same order for each of you. The
        race ends the moment either snake crashes — whoever has eaten more at that instant wins.
      </p>
      <Race />
    </main>
  );
}
