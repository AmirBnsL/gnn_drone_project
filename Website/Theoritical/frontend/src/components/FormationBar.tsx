type Props = {
  activeDigit: number | null;
  disabled: boolean;
  onSelect: (digit: number) => void;
};

export function FormationBar({ activeDigit, disabled, onSelect }: Props) {
  return (
    <footer className="formation-bar">
      {Array.from({ length: 10 }, (_, d) => (
        <button
          key={d}
          type="button"
          className={`btn ${activeDigit === d ? "active" : ""}`}
          disabled={disabled}
          onClick={() => onSelect(d)}
        >
          {d}
        </button>
      ))}
    </footer>
  );
}
