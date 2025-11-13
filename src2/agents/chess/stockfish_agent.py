import chess
import chess.engine

# Path to your Stockfish binary (download from https://stockfishchess.org/download/)
engine_path = "/usr/local/bin/stockfish"

# Create engine interface
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

# Create a board (default is starting position)
board = chess.Board()

# Get best move and evaluation
info = engine.analyse(board, chess.engine.Limit(time=0.1))
print("Best move:", info["pv"][0])
print("Evaluation:", info["score"])

# Clean up
engine.quit()
