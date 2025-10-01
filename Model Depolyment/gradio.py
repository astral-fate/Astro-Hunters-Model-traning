#!/usr/bin/env python3
"""
NASA Space Apps 2025 - FIXED Game Interface
Converts Astropy arrays to numpy before plotting
"""

import gradio as gr
import numpy as np
import pickle
import matplotlib.pyplot as plt
import traceback

# ============================================================================
# LOAD PREPROCESSED DATA
# ============================================================================
DATA_PATH = "preprocessed_tess_data.pkl"

try:
    with open(DATA_PATH, 'rb') as f:
        dataset = pickle.load(f)
    
    star_data = dataset['star_data']
    stars_available = list(star_data.keys())
    
    print(f"✅ Loaded {len(stars_available)} stars")
    DATA_LOADED = True

except Exception as e:
    print(f"❌ ERROR: {e}")
    DATA_LOADED = False
    stars_available = []
    star_data = {}

# ============================================================================
# GAME STATE
# ============================================================================
game_state = {
    'score': 0,
    'attempts': 0,
    'correct': 0,
    'current_star': None,
    'user_selections': [],
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_star_for_game(star_name):
    """Load star and create plot"""
    
    if not DATA_LOADED:
        return None, "ERROR: Data not loaded", ""
    
    try:
        data = star_data[star_name]
        
        # CRITICAL FIX: Convert to pure numpy arrays
        time = np.asarray(data['time']).ravel()
        flux = np.asarray(data['flux_detrended']).ravel()
        period = float(data['period'])
        
        print(f"Loaded {star_name}: time shape={time.shape}, flux shape={flux.shape}")
        
        # Calculate ground truth
        phase = (time % period) / period
        true_transits = (phase < 0.02) | (phase > 0.98)
        
        # Store in game state
        game_state['current_star'] = star_name
        game_state['time'] = time
        game_state['flux'] = flux
        game_state['true_transits'] = true_transits
        game_state['period'] = period
        game_state['user_selections'] = []
        
        # Create plot
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(time, flux, 'k.', markersize=2, alpha=0.4)
        ax.set_xlabel('Time (BTJD days)', fontsize=13)
        ax.set_ylabel('Normalized Brightness', fontsize=13)
        ax.set_title(f'{star_name} - Find the transit dips!', fontsize=15, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        instructions = f"""
        ## How to Play
        
        **Star:** {star_name}  
        **Period:** {period:.2f} days  
        
        1. Look for U-shaped dips in the plot above
        2. Note the time (x-axis) where dips occur
        3. Enter START and END time below
        4. Click "Add Selection"
        5. Repeat for 2-3 dips
        6. Click "Check Answer"
        
        **Hint:** Dips repeat every {period:.2f} days
        """
        
        return fig, instructions, f"✅ Loaded {star_name}"
        
    except Exception as e:
        error = f"ERROR: {str(e)}\n{traceback.format_exc()}"
        print(error)
        return None, error, "Error"

def add_selection(start, end):
    """Add user selection"""
    
    if game_state['current_star'] is None:
        return "⚠️ Load a star first"
    
    if start is None or end is None or start >= end:
        return "⚠️ Invalid time range"
    
    game_state['user_selections'].append((float(start), float(end)))
    return f"✅ Added selection #{len(game_state['user_selections'])}: {start:.1f} - {end:.1f}"

def check_answer():
    """Check user's answer"""
    
    if game_state['current_star'] is None:
        return None, "⚠️ Load a star first", ""
    
    if not game_state['user_selections']:
        return None, "⚠️ Add at least one selection", ""
    
    try:
        time = game_state['time']
        flux = game_state['flux']
        true_transits = game_state['true_transits']
        
        # Check overlap
        found = False
        correct_points = 0
        
        for (t_start, t_end) in game_state['user_selections']:
            mask = (time >= t_start) & (time <= t_end)
            overlap = np.sum(true_transits[mask])
            correct_points += overlap
            if overlap > 10:
                found = True
        
        # Score
        game_state['attempts'] += 1
        
        if found:
            game_state['correct'] += 1
            game_state['score'] += 100
            result = "🎉 CORRECT!"
            color = 'green'
        else:
            game_state['score'] = max(0, game_state['score'] - 20)
            result = "❌ Not quite"
            color = 'orange'
        
        # Plot result
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(time, flux, 'k.', markersize=2, alpha=0.3, label='Data')
        ax.scatter(time[true_transits], flux[true_transits], 
                  c='lime', s=15, alpha=0.7, label='Actual transits', zorder=5)
        
        for i, (t_start, t_end) in enumerate(game_state['user_selections']):
            ax.axvspan(t_start, t_end, alpha=0.3, color=color, 
                      label='Your selection' if i == 0 else '')
        
        ax.set_xlabel('Time (BTJD)', fontsize=13)
        ax.set_ylabel('Brightness', fontsize=13)
        ax.set_title(result, fontsize=15, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        stats = f"""
        ## {result}
        
        **Correct points:** {correct_points}  
        **Score:** {game_state['score']}  
        **Accuracy:** {game_state['correct']}/{game_state['attempts']} ({100*game_state['correct']/max(1,game_state['attempts']):.0f}%)
        """
        
        return fig, stats, f"Score: {game_state['score']}"
        
    except Exception as e:
        error = f"ERROR: {str(e)}\n{traceback.format_exc()}"
        print(error)
        return None, error, "Error"

# ============================================================================
# INTERFACE
# ============================================================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🔭 Exoplanet Transit Hunter\n### Find planets by detecting brightness dips!")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1️⃣ Choose Star")
            star_drop = gr.Dropdown(choices=stars_available, label="Star", 
                                   value=stars_available[0] if stars_available else None)
            load_btn = gr.Button("Load Star", variant="primary")
            
            gr.Markdown("### 2️⃣ Mark Dips")
            start_box = gr.Number(label="Start Time")
            end_box = gr.Number(label="End Time")
            add_btn = gr.Button("Add Selection")
            
            status = gr.Textbox(label="Status", lines=2)
            
            gr.Markdown("### 3️⃣ Check")
            check_btn = gr.Button("Check Answer", variant="primary")
        
        with gr.Column(scale=2):
            plot = gr.Plot()
            info = gr.Markdown()
    
    load_btn.click(load_star_for_game, star_drop, [plot, info, status])
    add_btn.click(add_selection, [start_box, end_box], status)
    check_btn.click(check_answer, outputs=[plot, info, status])

demo.launch(share=True, debug=True)
