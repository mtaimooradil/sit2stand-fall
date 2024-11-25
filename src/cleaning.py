import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
import numpy as np

data = np.load('./src/Python Notebooks/time_series_without_delta.npy', allow_pickle=True).item()

problematic_subjects_y_rankle = ['6HTqE41x', '15IPa5iS', 'AA9IzAL1', 'AVXJQX9B', 'bePyG1Du', 
                              'Cw8bz9tw', 'GESYi2xq', 'HA57ijek', 'iJsDlll8', 
                              'JoT31ute', 'PA4bXBlr', 'qUI9PiJS', 
                              'sUlatygS', 'UpNWCJ9a', 'xgN0wuHW', 'ztKJoXiw']

sub = problematic_subjects_y_rankle[15]

data = data[sub]['data'][:,11,1]

x = np.arange(len(data))
y = data

# Create a figure and axis
fig, ax = plt.subplots()
sc = ax.scatter(x, y, c='blue', s=5)


# Variables to store selected points and rectangles
selected_indices = []
rect_selector = None
rectangles = []  # Store rectangle patches
rect_coords = []  # Store rectangle coordinates for point deletion
selecting = True

# Rectangle selector functionality
def onselect(eclick, erelease):
    global selected_indices, rectangles, rect_coords

    # Get the rectangle boundaries
    x_min = min(eclick.xdata, erelease.xdata)
    x_max = max(eclick.xdata, erelease.xdata)
    y_min = min(eclick.ydata, erelease.ydata)
    y_max = max(eclick.ydata, erelease.ydata)

    # Add a rectangle to the plot
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor='green', linewidth=2)
    rectangles.append(rect)
    rect_coords.append([x_min, x_max, y_min, y_max])  # Store rectangle coordinates
    ax.add_patch(rect)

    # Find the points within the rectangle
    indices = np.where((x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max))[0]
    selected_indices.extend(indices)
    
    selected_points = np.c_[x[selected_indices], y[selected_indices]]
    print("Selected points:", selected_points)

    # Highlight selected points in red
    sc.set_color(['red' if i in selected_indices else 'blue' for i in range(len(x))])
    plt.draw()

# Save function to write selected points to a file
def save(event):
    if len(selected_indices) > 0:
        filename = f"{ax.get_title()}.txt"
        selected_points = np.c_[x[selected_indices], y[selected_indices]]
        np.savetxt(filename, selected_points, fmt='%f', header='x, y')
        print(f"Saved to {filename}")
    else:
        print("No points selected to save!")

# Undo last rectangle
def undo(event):
    global rectangles, rect_coords, selected_indices
    if rectangles:
        # Remove the last rectangle
        last_rect = rectangles.pop()
        last_rect.remove()  # Remove the visual rectangle from the plot
        last_coords = rect_coords.pop()

        # Remove points selected within the last rectangle
        x_min, x_max, y_min, y_max = last_coords
        indices_to_remove = np.where((x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max))[0]
        selected_indices = list(set(selected_indices) - set(indices_to_remove))  # Remove indices

        # Update point colors
        sc.set_color(['red' if i in selected_indices else 'blue' for i in range(len(x))])
        plt.draw()
    else:
        print("No more rectangles to undo")

# Delete a specific rectangle based on mouse click position
def delete(event):
    global rectangles, rect_coords, selected_indices

    if event.inaxes != ax:
        return  # Ignore clicks outside the plot

    x_click = event.xdata
    y_click = event.ydata

    for i, (x_min, x_max, y_min, y_max) in enumerate(rect_coords):
        # Check if click is inside the rectangle
        if x_min <= x_click <= x_max and y_min <= y_click <= y_max:
            # Remove the rectangle
            rectangles[i].remove()
            del rectangles[i]
            del_coords = rect_coords.pop(i)

            # Remove points selected within the deleted rectangle
            x_min, x_max, y_min, y_max = del_coords
            indices_to_remove = np.where((x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max))[0]
            selected_indices = list(set(selected_indices) - set(indices_to_remove))  # Remove indices

            # Update point colors
            sc.set_color(['red' if i in selected_indices else 'blue' for i in range(len(x))])
            plt.draw()
            break  # Delete only one rectangle

# Toggle between Select Mode and Zoom Mode
def toggle_mode(event):
    global selecting, rect_selector
    toolbar = plt.get_current_fig_manager().toolbar

    if selecting:
        # Deactivate the selector and enable zoom/pan mode
        rect_selector.set_active(False)
        toolbar.zoom()  # Activate zoom/pan mode
        btn_mode.label.set_text("Select")
        selecting = False
    else:
        # Deactivate zoom/pan mode and enable the selector
        toolbar.zoom()  # Deactivate zoom/pan mode
        rect_selector.set_active(True)
        btn_mode.label.set_text("Zoom")
        selecting = True

# Define button properties (common properties for better layout)
button_props = dict(width=0.12, height=0.05, borderpad=0.3)

# Add a button to save selected points
ax_button_save = plt.axes([0.8, 0.01, button_props['width'], button_props['height']])
btn_save = Button(ax_button_save, 'Save', hovercolor='0.975')
btn_save.on_clicked(save)

# Add a button to toggle between selection and zoom modes
ax_button_mode = plt.axes([0.65, 0.01, button_props['width'], button_props['height']])
btn_mode = Button(ax_button_mode, 'Zoom', hovercolor='0.975')
btn_mode.on_clicked(toggle_mode)

# Add an undo button to remove the last drawn rectangle
ax_button_undo = plt.axes([0.5, 0.01, button_props['width'], button_props['height']])
btn_undo = Button(ax_button_undo, 'Undo', hovercolor='0.975')
btn_undo.on_clicked(undo)

# Listen for clicks to delete a specific rectangle
fig.canvas.mpl_connect('button_press_event', delete)

# Set a title for the graph (this will be the filename)
ax.set_title(sub)

# Initialize RectangleSelector and set it active initially
rect_selector = RectangleSelector(ax, onselect, useblit=True, button=[1], interactive=True)
rect_selector.set_active(True)  # Starts in select mode

# Show the plot
plt.show()

