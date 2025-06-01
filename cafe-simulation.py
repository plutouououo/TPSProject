"""Simulasi Persebaran Cafe

Nama Anggota:
1. Ameliana Hardianti Utari 23/513968/TK/56455
2. Rahma Putri Anjani 23/519131/TK/57233
3. Fadel Aulia Naldi 23/519144/TK/57236
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import matplotlib.animation as animation

class CafeSimulation:
    """Simulasi persebaran kafe di Yogyakarta"""

    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.universities = [
            {'name': 'UGM', 'x': 3, 'y': 2},
            {'name': 'UPN', 'x': 1, 'y': 7},
            {'name': 'UNY', 'x': 6, 'y': 5}
        ]
        self.grid = None
        self.simulation_results = {}
        self.experiment_results = []

    def create_grid(self):
        """Buat grid sederhana dengan probabilitas kafe"""
        grid_data = []

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # Hitung jarak ke universitas terdekat
                min_distance = float('inf')
                for uni in self.universities:
                    distance = ((x - uni['x'])**2 + (y - uni['y'])**2)**0.5
                    min_distance = min(min_distance, distance)

                # Probabilitas tinggi jika dekat universitas
                if min_distance <= 1:
                    probability = 0.8
                elif min_distance <= 2:
                    probability = 0.6
                elif min_distance <= 3:
                    probability = 0.4
                else:
                    probability = 0.2

                probability += random.uniform(-0.1, 0.1)
                probability = max(0, min(1, probability))

                grid_data.append({
                    'x': x,
                    'y': y,
                    'distance_to_uni': round(min_distance, 2),
                    'probability': round(probability, 3),
                    'has_cafe': False
                })

        self.grid = pd.DataFrame(grid_data)
        return self.grid

    def simulate_cafes(self, threshold=0.5):
        """Simulasi pembukaan kafe berdasarkan probabilitas"""
        if self.grid is None:
            self.create_grid()

        self.grid['has_cafe'] = False

        for idx, row in self.grid.iterrows():
            if random.random() < row['probability']:
                self.grid.loc[idx, 'has_cafe'] = True

        total_cafes = self.grid['has_cafe'].sum()
        avg_probability = self.grid['probability'].mean()
        
        near_uni_cafes = len(self.grid[(self.grid['distance_to_uni'] <= 2) & (self.grid['has_cafe'])])
        far_uni_cafes = len(self.grid[(self.grid['distance_to_uni'] > 2) & (self.grid['has_cafe'])])

        self.simulation_results = {
            'total_cafes': total_cafes,
            'avg_probability': avg_probability,
            'near_uni_cafes': near_uni_cafes,
            'far_uni_cafes': far_uni_cafes,
            'grid_size': f"{self.grid_size}x{self.grid_size}"
        }
        return self.grid

    def get_recommendations(self):
        """Berikan rekomendasi lokasi terbaik"""
        if self.grid is None:
            self.create_grid()

        best_locations = self.grid[
            (self.grid['probability'] >= 0.6) &
            (~self.grid['has_cafe'])
        ].sort_values('probability', ascending=False).head(5)

        recommendations = []
        for i, (_, row) in enumerate(best_locations.iterrows(), 1):
            recommendations.append({
                'rank': i,
                'x': row['x'],
                'y': row['y'],
                'probability': row['probability'],
                'distance': row['distance_to_uni']
            })
        return recommendations

    def run_experiments(self, num_experiments=5):
        """Jalankan eksperimen multiple simulasi"""
        self.experiment_results = []
        for i in range(num_experiments):
            sim_exp = CafeSimulation(grid_size=self.grid_size)
            sim_exp.create_grid()
            sim_exp.simulate_cafes()
            total_cafes = sim_exp.grid['has_cafe'].sum()
            self.experiment_results.append(total_cafes)
        return self.experiment_results

    def simulate_cafes_stepwise(self, steps=None, seed=None):
        """Simulasi step-by-step untuk animasi"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        if self.grid is None:
            self.create_grid()
        
        grid = self.grid.copy()
        grid['has_cafe'] = False
        locs = grid.index.tolist()
        random.shuffle(locs)
        states = []
        cafes_opened = 0
        total = len(locs)
        
        for idx in locs:
            if random.random() < grid.at[idx, 'probability']:
                grid.at[idx, 'has_cafe'] = True
                cafes_opened += 1
            states.append(grid.copy())
            if cafes_opened >= steps:
                break
        return states

    def animate_video(self, filename='cafe_simulation.gif', 
                      steps=None, interval=300, fps=15, seed=None):
        print("Memulai simulasi...")
        
        self.create_grid()
        self.simulate_cafes()
        self.run_experiments()
        
        recommendations = self.get_recommendations()[:3]
        cafe_snapshots = self.simulate_cafes_stepwise(steps=steps, seed=seed)
        
        fig = plt.figure(figsize=(16, 10))
        
        gs = fig.add_gridspec(2, 3, width_ratios=[2, 1, 1], height_ratios=[2, 1])
                
        ax_main = fig.add_subplot(gs[0, 0])
        ax_info = fig.add_subplot(gs[0, 1:])
        ax_heatmap1 = fig.add_subplot(gs[1, 0])
        ax_heatmap2 = fig.add_subplot(gs[1, 1])
        ax_stats = fig.add_subplot(gs[1, 2])
        
        def plot_main_animation(grid_data, step_num):
            """Plot animasi utama"""
            ax_main.clear()
            
            for _, row in grid_data.iterrows():
                color = 'red' if row['has_cafe'] else 'lightblue'
                size = 120 if row['has_cafe'] else 40
                ax_main.scatter(row['x'], row['y'], c=color, s=size, alpha=0.7)
            
            for uni in self.universities:
                ax_main.scatter(uni['x'], uni['y'], c='gold', s=250, marker='*',
                              edgecolors='black', linewidth=2)
                ax_main.text(uni['x'], uni['y']+0.4, uni['name'], 
                           ha='center', fontweight='bold', fontsize=10)
            
            current_cafes = grid_data['has_cafe'].sum()
            ax_main.text(0.02, 0.98, f'Step: {step_num}\nKafe: {current_cafes}', 
                        transform=ax_main.transAxes, fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
            
            ax_main.set_title('SIMULASI PENYEBARAN KAFE YOGYAKARTA', 
                             fontsize=14, fontweight='bold', pad=10)
            ax_main.set_xlim(-0.5, self.grid_size-0.5)
            ax_main.set_ylim(-0.5, self.grid_size-0.5)
            ax_main.grid(True, alpha=0.3)
            ax_main.set_xlabel('Koordinat X')
            ax_main.set_ylabel('Koordinat Y')
        
        def plot_info_panel():
            """Plot panel informasi"""
            ax_info.clear()
            ax_info.axis('off')
            
            info_text = f"""HASIL SIMULASI
Total Kafe: {self.simulation_results['total_cafes']}
Avg Probability: {self.simulation_results['avg_probability']:.3f}
Grid Size: {self.simulation_results['grid_size']}

PERSEBARAN
Dekat Univ (≤2): {self.simulation_results['near_uni_cafes']}
Jauh Univ (>2): {self.simulation_results['far_uni_cafes']}

TOP 3 REKOMENDASI"""
            
            y_pos = 0.95
            for line in info_text.split('\n'):
                if line.strip():
                    weight = 'bold' if line.startswith(('📊', '📍', '🎯')) else 'normal'
                    ax_info.text(0.05, y_pos, line, transform=ax_info.transAxes, 
                               fontsize=9, fontweight=weight, ha='left', va='top')
                y_pos -= 0.08
            
            for i, rec in enumerate(recommendations):
                rec_text = f"{i+1}. ({rec['x']},{rec['y']}) P:{rec['probability']:.2f}"
                ax_info.text(0.05, y_pos, rec_text, transform=ax_info.transAxes,
                           fontsize=8, ha='left', va='top', fontfamily='monospace')
                y_pos -= 0.06
        
        def plot_heatmaps():
            """Plot heatmaps"""
            ax_heatmap1.clear()
            data_prob = self.grid.pivot_table(index='y', columns='x', values='probability')
            sns.heatmap(data_prob, annot=True, fmt='.2f', cmap='YlOrRd', 
                       cbar=False, ax=ax_heatmap1, annot_kws={'size': 7})
            ax_heatmap1.set_title('Probability', fontsize=10, fontweight='bold')
            ax_heatmap1.set_xlabel('X', fontsize=8)
            ax_heatmap1.set_ylabel('Y', fontsize=8)
            ax_heatmap1.tick_params(labelsize=7)
            
            ax_heatmap2.clear()
            current_grid = cafe_snapshots[min(len(cafe_snapshots)-1, max(0, len(cafe_snapshots)//2))]
            data_cafe = current_grid.copy()
            data_cafe['has_cafe'] = data_cafe['has_cafe'].astype(int)
            data_cafe = data_cafe.pivot_table(index='y', columns='x', values='has_cafe')
            sns.heatmap(data_cafe, annot=True, fmt='.0f', cmap='Blues', 
                       cbar=False, ax=ax_heatmap2, annot_kws={'size': 7})
            ax_heatmap2.set_title('Kafe Locations', fontsize=10, fontweight='bold')
            ax_heatmap2.set_xlabel('X', fontsize=8)
            ax_heatmap2.set_ylabel('Y', fontsize=8)
            ax_heatmap2.tick_params(labelsize=7)
        
        def plot_experiment_stats():
            """Plot hasil eksperimen"""
            ax_stats.clear()
            ax_stats.axis('off')
            
            if self.experiment_results:
                stats_text = f"""EKSPERIMEN ({len(self.experiment_results)} Simulasi) Hasil:"""
                y_pos = 0.9
                ax_stats.text(0.05, y_pos, stats_text, transform=ax_stats.transAxes,
                            fontsize=9, fontweight='bold', ha='left', va='top')
                y_pos -= 0.15
                
                for i, result in enumerate(self.experiment_results, 1):
                    exp_text = f"Sim {i}: {result}"
                    ax_stats.text(0.05, y_pos, exp_text, transform=ax_stats.transAxes,
                                fontsize=8, ha='left', va='top', fontfamily='monospace')
                    y_pos -= 0.1
                
                avg_result = np.mean(self.experiment_results)
                range_text = f"\nAvg: {avg_result:.1f}\nRange: {min(self.experiment_results)}-{max(self.experiment_results)}"
                ax_stats.text(0.05, y_pos-0.05, range_text, transform=ax_stats.transAxes,
                            fontsize=8, ha='left', va='top', fontweight='bold')
        
        def update(frame_idx):
            if frame_idx < len(cafe_snapshots):
                current_snapshot = cafe_snapshots[frame_idx]
                plot_main_animation(current_snapshot, frame_idx + 1)
                plot_info_panel()
                plot_heatmaps()
                plot_experiment_stats()
                
                ax_heatmap2.clear()
                data_cafe = current_snapshot.copy()
                data_cafe['has_cafe'] = data_cafe['has_cafe'].astype(int)
                data_cafe = data_cafe.pivot_table(index='y', columns='x', values='has_cafe')
                sns.heatmap(data_cafe, annot=True, fmt='.0f', cmap='Blues', 
                           cbar=False, ax=ax_heatmap2, annot_kws={'size': 7})
                ax_heatmap2.set_title(f'Kafe (Step {frame_idx+1})', fontsize=10, fontweight='bold')
                ax_heatmap2.set_xlabel('X', fontsize=8)
                ax_heatmap2.set_ylabel('Y', fontsize=8)
                ax_heatmap2.tick_params(labelsize=7)
        
        ani = animation.FuncAnimation(fig, update, frames=len(cafe_snapshots), 
                                    interval=interval, blit=False, repeat=True)
        
        # Simpan sebagai GIF
        print(f"🎬 Menyimpan GIF ke {filename} ...")
        ani.save(filename, writer='pillow', fps=fps)
        plt.close(fig)
        print("✅ GIF berhasil disimpan!")
        return None

if __name__ == "__main__":
    print("SIMULASI KAFE YOGYAKARTA")
    print("=" * 60)
    sim = CafeSimulation(grid_size=8)
    
    print("\n🎬 Membuat animasi GIF...")
    sim.animate_video(
        filename='cafe_simulation.gif',
        steps=30,
        interval=500,
        fps=12,
        seed=42
    )
    print("\n✅ Simulasi selesai!")
    print("File output: cafe_simulation.gif")