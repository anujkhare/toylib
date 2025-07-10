from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS

# Import the game classes (assuming gametypes.py is in the same directory)
from toylib_projects.dotsAndBoxes.gametypes import Game

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Store game instances (in production, use a proper database)
games = {}
next_game_id = 1

# HTML template for the frontend
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Dots and Boxes</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background-color: #f0f0f0;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .game-info {
            margin-bottom: 20px;
            text-align: center;
        }
        .scores {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 10px 0;
        }
        .score {
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
        }
        .player1 { background-color: #ff6b6b; color: white; }
        .player2 { background-color: #4ecdc4; color: white; }
        .current-turn { box-shadow: 0 0 10px rgba(0,0,0,0.3); }
        .game-board {
            position: relative;
            margin: 20px auto;
        }
        .dot {
            width: 10px;
            height: 10px;
            background-color: #333;
            border-radius: 50%;
            position: absolute;
            z-index: 3;
        }
        .line {
            position: absolute;
            background-color: #ddd;
            cursor: pointer;
            transition: all 0.3s;
            z-index: 1;
        }
        .line:hover:not(.filled) {
            background-color: #999;
        }
        .line.filled {
            cursor: default;
            background-color: #333 !important;
        }
        .line.filled.player1 { 
            background-color: #ff6b6b !important;
            box-shadow: 0 0 5px rgba(255, 107, 107, 0.5);
        }
        .line.filled.player2 { 
            background-color: #4ecdc4 !important;
            box-shadow: 0 0 5px rgba(78, 205, 196, 0.5);
        }
        .horizontal {
            height: 6px;
            width: 50px;
        }
        .horizontal.filled {
            height: 8px;
            margin-top: -1px;
        }
        .vertical {
            width: 6px;
            height: 50px;
        }
        .vertical.filled {
            width: 8px;
            margin-left: -1px;
        }
        .box {
            position: absolute;
            width: 50px;
            height: 50px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 20px;
            color: white;
            z-index: 0;
        }
        .box.player1 { background-color: rgba(255, 107, 107, 0.3); }
        .box.player2 { background-color: rgba(78, 205, 196, 0.3); }
        .controls {
            text-align: center;
            margin-top: 20px;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            background-color: #333;
            color: white;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #555;
        }
        .game-over {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Dots and Boxes</h1>
        <div class="game-info">
            <div id="game-status"></div>
            <div class="scores">
                <div class="score player1" id="score1">Player 1: 0</div>
                <div class="score player2" id="score2">Player 2: 0</div>
            </div>
        </div>
        <div id="game-board" class="game-board"></div>
        <div class="controls">
            <button onclick="newGame()">New Game</button>
        </div>
        <div id="game-over" class="game-over"></div>
    </div>

    <script>
        let gameId = null;
        let gameState = null;
        const DOT_SIZE = 10;
        const LINE_LENGTH = 50;
        const SPACING = 60;

        async function newGame() {
            const rows = 4;
            const cols = 4;
            
            const response = await fetch('/api/game/new', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({rows, cols})
            });
            
            const data = await response.json();
            gameId = data.game_id;
            await updateGame();
        }

        async function makeMove(action) {
            if (!gameId) return;
            
            console.log('Making move:', action);
            
            try {
                const response = await fetch(`/api/game/${gameId}/move`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action})
                });
                
                if (response.ok) {
                    const result = await response.json();
                    console.log('Move successful:', result);
                    await updateGame();
                } else {
                    const error = await response.json();
                    console.error('Move failed:', error);
                    alert(error.error);
                }
            } catch (error) {
                console.error('Error making move:', error);
                alert('Error making move: ' + error.message);
            }
        }

        async function updateGame() {
            if (!gameId) return;
            
            try {
                const response = await fetch(`/api/game/${gameId}/state`);
                const data = await response.json();
                console.log('Game state updated:', data);
                gameState = data;
                renderGame();
            } catch (error) {
                console.error('Error updating game:', error);
                alert('Error updating game: ' + error.message);
            }
        }

        function renderGame() {
            const board = document.getElementById('game-board');
            board.innerHTML = '';
            
            const rows = gameState.rows;
            const cols = gameState.cols;
            
            // Set board size
            board.style.width = (cols - 1) * SPACING + DOT_SIZE + 'px';
            board.style.height = (rows - 1) * SPACING + DOT_SIZE + 'px';
            
            // Draw dots
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    const dot = document.createElement('div');
                    dot.className = 'dot';
                    dot.style.left = c * SPACING + 'px';
                    dot.style.top = r * SPACING + 'px';
                    board.appendChild(dot);
                }
            }
            
            // Draw horizontal lines
            let actionIndex = 0;
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols - 1; c++) {
                    const line = document.createElement('div');
                    line.className = 'line horizontal';
                    line.style.left = c * SPACING + DOT_SIZE + 'px';
                    line.style.top = r * SPACING + 2 + 'px';
                    
                    const hActionIndex = (rows - 1) * cols + r * (cols - 1) + c;
                    
                    if (gameState.filled_horizontal[r][c]) {
                        line.classList.add('filled');
                        // Determine which player filled this line
                        const player = getLineOwner(r, c, 'horizontal');
                        if (player !== null) {
                            line.classList.add('player' + (player + 1));
                        }
                    } else {
                        line.onclick = () => makeMove(hActionIndex);
                    }
                    
                    board.appendChild(line);
                }
            }
            
            // Draw vertical lines
            for (let r = 0; r < rows - 1; r++) {
                for (let c = 0; c < cols; c++) {
                    const line = document.createElement('div');
                    line.className = 'line vertical';
                    line.style.left = c * SPACING + 2 + 'px';
                    line.style.top = r * SPACING + DOT_SIZE + 'px';
                    
                    const vActionIndex = r * cols + c;
                    
                    if (gameState.filled_vertical[r][c]) {
                        line.classList.add('filled');
                        // Determine which player filled this line
                        const player = getLineOwner(r, c, 'vertical');
                        if (player !== null) {
                            line.classList.add('player' + (player + 1));
                        }
                    } else {
                        line.onclick = () => makeMove(vActionIndex);
                    }
                    
                    board.appendChild(line);
                }
            }
            
            // Draw completed boxes
            for (let r = 0; r < rows - 1; r++) {
                for (let c = 0; c < cols - 1; c++) {
                    if (gameState.boxes_by_player[0][r][c] || gameState.boxes_by_player[1][r][c]) {
                        const box = document.createElement('div');
                        box.className = 'box';
                        box.style.left = c * SPACING + DOT_SIZE + 'px';
                        box.style.top = r * SPACING + DOT_SIZE + 'px';
                        
                        if (gameState.boxes_by_player[0][r][c]) {
                            box.classList.add('player1');
                            box.textContent = '1';
                        } else {
                            box.classList.add('player2');
                            box.textContent = '2';
                        }
                        
                        board.appendChild(box);
                    }
                }
            }
            
            // Update scores and game status
            const score1 = gameState.boxes_by_player[0].flat().filter(x => x).length;
            const score2 = gameState.boxes_by_player[1].flat().filter(x => x).length;
            
            document.getElementById('score1').textContent = `Player 1: ${score1}`;
            document.getElementById('score2').textContent = `Player 2: ${score2}`;
            
            // Update current player highlight
            document.getElementById('score1').classList.toggle('current-turn', gameState.next_player === 0);
            document.getElementById('score2').classList.toggle('current-turn', gameState.next_player === 1);
            
            // Check if game is over
            const totalBoxes = (rows - 1) * (cols - 1);
            if (score1 + score2 === totalBoxes) {
                let message = '';
                if (score1 > score2) {
                    message = 'Player 1 Wins!';
                } else if (score2 > score1) {
                    message = 'Player 2 Wins!';
                } else {
                    message = "It's a Tie!";
                }
                document.getElementById('game-over').textContent = message;
                document.getElementById('game-status').textContent = 'Game Over';
            } else {
                document.getElementById('game-over').textContent = '';
                document.getElementById('game-status').textContent = `Player ${gameState.next_player + 1}'s Turn`;
            }
        }

        function getLineOwner(r, c, direction) {
            // This is a simplified version - in a real game, you'd track who drew each line
            // For now, we'll color lines based on who owns adjacent boxes
            if (direction === 'horizontal') {
                const boxAbove = r > 0 && gameState.boxes_by_player[0][r-1][c] ? 0 : 
                               (r > 0 && gameState.boxes_by_player[1][r-1][c] ? 1 : null);
                const boxBelow = r < gameState.rows - 1 && gameState.boxes_by_player[0][r][c] ? 0 :
                               (r < gameState.rows - 1 && gameState.boxes_by_player[1][r][c] ? 1 : null);
                return boxAbove !== null ? boxAbove : boxBelow;
            } else {
                const boxLeft = c > 0 && gameState.boxes_by_player[0][r][c-1] ? 0 :
                              (c > 0 && gameState.boxes_by_player[1][r][c-1] ? 1 : null);
                const boxRight = c < gameState.cols - 1 && gameState.boxes_by_player[0][r][c] ? 0 :
                               (c < gameState.cols - 1 && gameState.boxes_by_player[1][r][c] ? 1 : null);
                return boxLeft !== null ? boxLeft : boxRight;
            }
        }

        // Start a new game when the page loads
        newGame();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Serve the game UI"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/game/new', methods=['POST'])
def new_game():
    """Create a new game instance"""
    global next_game_id
    
    data = request.json
    rows = data.get('rows', 4)
    cols = data.get('cols', 4)
    
    game = Game(rows, cols)
    game_id = next_game_id
    games[game_id] = game
    next_game_id += 1
    
    return jsonify({
        'game_id': game_id,
        'rows': rows,
        'cols': cols
    })

@app.route('/api/game/<int:game_id>/state', methods=['GET'])
def get_game_state(game_id):
    """Get the current state of a game"""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    game = games[game_id]
    state_dict = game.state.to_dict()
    
    return jsonify(state_dict)

@app.route('/api/game/<int:game_id>/move', methods=['POST'])
def make_move(game_id):
    """Make a move in the game"""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    game = games[game_id]
    data = request.json
    action = data.get('action')
    
    if action is None:
        return jsonify({'error': 'No action provided'}), 400
    
    try:
        # Check if action is valid
        if action not in game.valid_actions:
            return jsonify({'error': 'Invalid action'}), 400
        
        # Make the move (our patched version handles turn continuation correctly)
        game.move(action)
        
        return jsonify({
            'success': True,
            'next_player': game.state.next_player
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/game/<int:game_id>/valid_actions', methods=['GET'])
def get_valid_actions(game_id):
    """Get all valid actions for the current game state"""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    game = games[game_id]
    return jsonify({
        'valid_actions': game.valid_actions.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)