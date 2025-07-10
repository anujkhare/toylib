from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import numpy as np

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
            max-width: 800px;
            width: 100%;
        }
        .game-setup {
            margin-bottom: 20px;
            text-align: center;
        }
        .game-setup label {
            margin: 0 10px;
            font-weight: bold;
        }
        .game-setup input[type="number"] {
            width: 60px;
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 3px;
            margin: 0 5px;
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
            overflow-x: auto;
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
            margin: 0 5px;
        }
        button:hover {
            background-color: #555;
        }
        .game-over {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            background-color: #e8f5e8;
            color: #2d5a2d;
        }
        .game-over.tie {
            background-color: #fff8dc;
            color: #8b7355;
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Dots and Boxes</h1>
        
        <div class="game-setup">
            <label for="rows">Rows:</label>
            <input type="number" id="rows" min="2" max="10" value="4">
            
            <label for="cols">Columns:</label>
            <input type="number" id="cols" min="2" max="10" value="4">
            
            <button onclick="newGame()">New Game</button>
        </div>
        
        <div id="game-content" class="hidden">
            <div class="game-info">
                <div id="game-status"></div>
                <div class="scores">
                    <div class="score player1" id="score1">Player 1: 0</div>
                    <div class="score player2" id="score2">Player 2: 0</div>
                </div>
            </div>
            <div id="game-board" class="game-board"></div>
            <div id="game-over" class="game-over hidden"></div>
        </div>
    </div>

    <script>
        let gameId = null;
        let gameData = null;
        const DOT_SIZE = 10;
        const LINE_LENGTH = 50;
        const SPACING = 60;

        async function newGame() {
            const rows = parseInt(document.getElementById('rows').value);
            const cols = parseInt(document.getElementById('cols').value);
            
            if (rows < 2 || cols < 2 || rows > 10 || cols > 10) {
                alert('Please enter valid dimensions (2-10 for both rows and columns)');
                return;
            }
            
            try {
                const response = await fetch('/api/game/new', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({rows, cols})
                });
                
                if (!response.ok) {
                    throw new Error('Failed to create new game');
                }
                
                const data = await response.json();
                gameId = data.game_id;
                document.getElementById('game-content').classList.remove('hidden');
                await updateGame();
            } catch (error) {
                console.error('Error creating new game:', error);
                alert('Error creating new game: ' + error.message);
            }
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
                    console.log('Move result:', result);
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
                const response = await fetch(`/api/game/${gameId}/full_state`);
                if (!response.ok) {
                    throw new Error('Failed to get game state');
                }
                
                const data = await response.json();
                console.log('Game state updated:', data);
                gameData = data;
                renderGame();
            } catch (error) {
                console.error('Error updating game:', error);
                alert('Error updating game: ' + error.message);
            }
        }

        function renderGame() {
            if (!gameData) return;
            
            const board = document.getElementById('game-board');
            board.innerHTML = '';
            
            const state = gameData.state;
            const lineOwners = gameData.line_owners;
            const gameInfo = gameData.game_info;
            
            const rows = state.rows;
            const cols = state.cols;
            
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
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols - 1; c++) {
                    const line = document.createElement('div');
                    line.className = 'line horizontal';
                    line.style.left = c * SPACING + DOT_SIZE + 'px';
                    line.style.top = r * SPACING + 2 + 'px';
                    
                    // Calculate action index for horizontal line
                    const hActionIndex = (rows - 1) * cols + r * (cols - 1) + c;
                    
                    if (state.filled_horizontal[r][c]) {
                        line.classList.add('filled');
                        // Use line owner information
                        const owner = lineOwners.horizontal[r][c];
                        if (owner >= 0) {
                            line.classList.add('player' + (owner + 1));
                        }
                    } else if (!gameInfo.game_over) {
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
                    
                    // Calculate action index for vertical line
                    const vActionIndex = r * cols + c;
                    
                    if (state.filled_vertical[r][c]) {
                        line.classList.add('filled');
                        // Use line owner information
                        const owner = lineOwners.vertical[r][c];
                        if (owner >= 0) {
                            line.classList.add('player' + (owner + 1));
                        }
                    } else if (!gameInfo.game_over) {
                        line.onclick = () => makeMove(vActionIndex);
                    }
                    
                    board.appendChild(line);
                }
            }
            
            // Draw completed boxes
            for (let r = 0; r < rows - 1; r++) {
                for (let c = 0; c < cols - 1; c++) {
                    if (state.boxes_by_player[0][r][c] || state.boxes_by_player[1][r][c]) {
                        const box = document.createElement('div');
                        box.className = 'box';
                        box.style.left = c * SPACING + DOT_SIZE + 'px';
                        box.style.top = r * SPACING + DOT_SIZE + 'px';
                        
                        if (state.boxes_by_player[0][r][c]) {
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
            document.getElementById('score1').textContent = `Player 1: ${gameInfo.scores.player1}`;
            document.getElementById('score2').textContent = `Player 2: ${gameInfo.scores.player2}`;
            
            // Update current player highlight
            document.getElementById('score1').classList.toggle('current-turn', gameInfo.current_player === 0);
            document.getElementById('score2').classList.toggle('current-turn', gameInfo.current_player === 1);
            
            // Handle game over state
            const gameOverDiv = document.getElementById('game-over');
            if (gameInfo.game_over) {
                let message = '';
                let className = 'game-over';
                
                if (gameInfo.winner === 0) {
                    message = `🎉 Player 1 Wins! (${gameInfo.scores.player1} - ${gameInfo.scores.player2})`;
                } else if (gameInfo.winner === 1) {
                    message = `🎉 Player 2 Wins! (${gameInfo.scores.player2} - ${gameInfo.scores.player1})`;
                } else {
                    message = `🤝 It's a Tie! (${gameInfo.scores.player1} - ${gameInfo.scores.player2})`;
                    className += ' tie';
                }
                
                gameOverDiv.textContent = message;
                gameOverDiv.className = className;
                document.getElementById('game-status').textContent = 'Game Over';
            } else {
                gameOverDiv.className = 'game-over hidden';
                document.getElementById('game-status').textContent = `Player ${gameInfo.current_player + 1}'s Turn`;
            }
        }

        // Initialize the page
        document.addEventListener('DOMContentLoaded', function() {
            // Focus on the new game button
            document.getElementById('rows').focus();
        });
    </script>
</body>
</html>
'''

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

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
    
    # Validate input
    if not isinstance(rows, int) or not isinstance(cols, int):
        return jsonify({'error': 'Rows and columns must be integers'}), 400
    
    if rows < 2 or cols < 2 or rows > 10 or cols > 10:
        return jsonify({'error': 'Rows and columns must be between 2 and 10'}), 400
    
    try:
        game = Game(rows, cols)
        game_id = next_game_id
        games[game_id] = game
        next_game_id += 1
        
        return jsonify({
            'game_id': game_id,
            'rows': rows,
            'cols': cols,
            'message': f'New {rows}x{cols} game created successfully'
        })
    except Exception as e:
        return jsonify({'error': f'Failed to create game: {str(e)}'}), 500

@app.route('/api/game/<int:game_id>/state', methods=['GET'])
def get_game_state(game_id):
    """Get the current state of a game"""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    game = games[game_id]
    state_dict = game.state.to_dict()
    
    return jsonify(convert_numpy_types(state_dict))

@app.route('/api/game/<int:game_id>/full_state', methods=['GET'])
def get_full_game_state(game_id):
    """Get the complete game state including line owners and game info"""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    game = games[game_id]
    full_state = convert_numpy_types(game.to_dict())
    return jsonify(full_state)

@app.route('/api/game/<int:game_id>/info', methods=['GET'])
def get_game_info(game_id):
    """Get game information summary"""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    game = games[game_id]
    info = convert_numpy_types(game.get_game_info())
    return jsonify(info)

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
    
    if not isinstance(action, int):
        return jsonify({'error': 'Action must be an integer'}), 400
    
    try:
        # Check if game is already over
        if game.state.is_game_over():
            return jsonify({'error': 'Game is already over'}), 400
        
        # Check if action is valid
        if action not in game.valid_actions:
            return jsonify({'error': 'Invalid action'}), 400
        
        # Make the move and get detailed result
        move_result = game.move(action)
        
        # Convert numpy types in move result
        move_result = convert_numpy_types(move_result)
        
        return jsonify({
            'success': True,
            'move_result': move_result
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
        'valid_actions': game.valid_actions.tolist(),
        'count': len(game.valid_actions)
    })

@app.route('/api/games', methods=['GET'])
def list_games():
    """Get list of all active games"""
    game_list = []
    for game_id, game in games.items():
        info = convert_numpy_types(game.get_game_info())
        game_list.append({
            'game_id': game_id,
            'rows': info['rows'],
            'cols': info['cols'],
            'game_over': info['game_over'],
            'scores': info['scores']
        })
    
    return jsonify({'games': game_list, 'count': len(game_list)})

@app.route('/api/game/<int:game_id>', methods=['DELETE'])
def delete_game(game_id):
    """Delete a game instance"""
    if game_id not in games:
        return jsonify({'error': 'Game not found'}), 404
    
    del games[game_id]
    return jsonify({'success': True, 'message': f'Game {game_id} deleted'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)