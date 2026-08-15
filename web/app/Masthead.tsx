import Link from 'next/link';

export function Masthead() {
  return (
    <header className="masthead">
      <Link href="/" className="wordmark">
        SNAKE<span>/</span>AGENT
      </Link>
      <nav className="nav">
        <Link href="/race">Race</Link>
        <Link href="/design">Design</Link>
        <Link href="/gallery">Gallery</Link>
        <Link href="/leaderboard">Leaderboard</Link>
      </nav>
    </header>
  );
}
