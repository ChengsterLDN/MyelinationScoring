import sys
import os
import glob
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageOps
from readlif.reader import LifFile
import numpy as np
from matplotlib import pyplot
import matplotlib.colors as mcolors
import re

# Semantic roles we need assigned to channels. "skip" means "don't extract this channel".
CHANNEL_ROLES = ["nuclei", "mbp", "pillar", "skip"]


class LIFProcessor:
    """Class to process LIF files and extract images as PNGs"""

    def __init__(self):
        # Fallback only, used to pre-fill the confirmation dialog's first guess.
        # This is NEVER used to silently label data - see resolve_channel_mapping().
        self.default_channel_guess = {0: "nuclei", 1: "mbp", 2: "pillar"}
        self.yellow_cmap = mcolors.LinearSegmentedColormap.from_list('yellow_cmap', ['black', 'yellow'])
        self.cyan_cmap = mcolors.LinearSegmentedColormap.from_list('cyan_cmap', ['black', 'cyan'])
        # Cache of confirmed mappings, keyed by a "channel signature" (LUT names etc.)
        # so the user is only asked once per distinct acquisition setup, not once per series.
        self.confirmed_mappings = {}
        self._tk_root = None  # lazily-created hidden root used to host dialogs
    
    def clean_filename(self, filename):
        """Clean filename by removing or replacing invalid characters"""
        # Remove trailing spaces
        filename = filename.strip()
        # Replace invalid characters with underscores
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Replace multiple spaces with single space
        filename = re.sub(r'\s+', ' ', filename)
        return filename
    
    # ------------------------------------------------------------------
    # Channel identification: extract metadata hints + confirm with user
    # ------------------------------------------------------------------

    def _collect_xml_image_elements(self, tree, return_list=None):
        """
        Walk the LIF XML tree in exactly the same order readlif uses internally
        (see LifFile._recursive_image_find) so that return_list[i] is the raw
        <Element> for lif_file.image_list[i] / the i-th image from get_iter_image().
        We do this ourselves (rather than touching readlif internals) so we can
        read metadata readlif doesn't expose, like per-channel LUTName/dye info.
        """
        if return_list is None:
            return_list = []

        children = tree.findall("./Children/Element")
        if len(children) < 1:
            children = tree.findall("./Element")

        for item in children:
            has_sub_children = len(item.findall("./Children/Element/Data")) > 0
            is_image = len(item.findall("./Data/Image")) > 0

            if is_image:
                return_list.append(item)
            if has_sub_children:
                self._collect_xml_image_elements(item, return_list)

        return return_list

    def get_channel_hints(self, lif_file, img_index):
        """
        Return a list of hint strings, one per channel, e.g. ['Blue', 'Green', 'Gray'].
        These come straight from the Leica-assigned LUTName / measured-quantity metadata
        and are a HINT for the user to check against, not a label we trust blindly -
        the same LUT color can be reused for different stains across labs/protocols.
        """
        try:
            xml_elements = self._collect_xml_image_elements(lif_file.xml_root)
            item = xml_elements[img_index]
            channel_descs = item.findall(
                "./Data/Image/ImageDescription/Channels/ChannelDescription"
            )
            hints = []
            for c in channel_descs:
                lut = c.attrib.get("LUTName", "")
                quantity = c.attrib.get("NameOfMeasuredQuantity", "")
                hint = " / ".join(x for x in [lut, quantity] if x) or "unknown"
                hints.append(hint)
            return hints
        except Exception as e:
            print(f"  Could not read channel metadata hints: {e}")
            return []

    def _make_preview_thumbnail(self, lif_image, channel, size=110):
        """Grab a representative frame for a channel and return it as a small PIL image."""
        try:
            z_mid = (lif_image.dims.z if hasattr(lif_image.dims, 'z') else 1) // 2
            pil_frame = lif_image.get_frame(t=0, z=z_mid, c=channel)
            frame = pil_frame.convert("L")
            frame = ImageOps.autocontrast(frame)
            frame.thumbnail((size, size))
            return frame
        except Exception as e:
            print(f"  Could not build preview for channel {channel}: {e}")
            return Image.new("L", (size, size), color=40)

    def _get_root(self):
        if self._tk_root is None:
            self._tk_root = tk.Tk()
            self._tk_root.withdraw()
        return self._tk_root

    def confirm_channel_mapping(self, series_name, n_channels, hints, lif_image):
        """
        Show a dialog with a thumbnail + metadata hint per channel and let the user
        assign each one to nuclei/mbp/pillar/skip. Returns {channel_index: role}.
        This is the step that actually guarantees correctness instead of assuming order.
        """
        root = self._get_root()
        dialog = tk.Toplevel(root)
        dialog.title(f"Confirm channel assignment - {series_name}")
        dialog.grab_set()

        tk.Label(
            dialog,
            text=(f"Series: {series_name}\n"
                  "Check each channel's preview/metadata hint and assign its identity.\n"
                  "This mapping will be reused for other series with the same channel setup."),
            justify="left"
        ).grid(row=0, column=0, columnspan=n_channels, padx=10, pady=(10, 5), sticky="w")

        result = {}
        photo_refs = []  # keep references alive, tkinter needs this
        role_vars = []

        for c in range(n_channels):
            col_frame = tk.Frame(dialog, bd=1, relief="groove", padx=6, pady=6)
            col_frame.grid(row=1, column=c, padx=6, pady=6)

            thumb = self._make_preview_thumbnail(lif_image, c)
            photo = ImageTk.PhotoImage(thumb)
            photo_refs.append(photo)
            tk.Label(col_frame, image=photo).pack()

            hint_text = hints[c] if c < len(hints) else "unknown"
            tk.Label(col_frame, text=f"Channel {c}\nLUT/dye hint: {hint_text}",
                     justify="center").pack(pady=(4, 4))

            guess = self.default_channel_guess.get(c, "skip")
            if guess not in CHANNEL_ROLES:
                guess = "skip"
            var = tk.StringVar(value=guess)
            role_vars.append(var)
            ttk.OptionMenu(col_frame, var, guess, *CHANNEL_ROLES).pack()

        def on_confirm():
            for c, var in enumerate(role_vars):
                result[c] = var.get()
            dialog.destroy()

        tk.Button(dialog, text="Confirm mapping", command=on_confirm).grid(
            row=2, column=0, columnspan=n_channels, pady=10
        )

        dialog.protocol("WM_DELETE_WINDOW", on_confirm)
        root.wait_window(dialog)
        return result

    def resolve_channel_mapping(self, lif_file, img_index, series_name, lif_image):
        """
        Return {channel_index: role} for this image, asking the user to confirm
        whenever we encounter a channel setup (LUT signature) we haven't seen yet
        in this run. Same signature -> reuse the previously confirmed mapping.
        """
        n_channels = lif_image.channels
        hints = self.get_channel_hints(lif_file, img_index)
        # Pad hints if metadata was missing/short so signature stays well-defined
        hints = (hints + ["unknown"] * n_channels)[:n_channels]
        signature = tuple(hints)

        if signature not in self.confirmed_mappings:
            print(f"  New channel setup detected {signature} - asking for confirmation")
            mapping = self.confirm_channel_mapping(series_name, n_channels, hints, lif_image)
            self.confirmed_mappings[signature] = mapping
        else:
            mapping = self.confirmed_mappings[signature]

        return mapping

    def browse_lif_file(self):
        """Open file dialog to browse for LIF files"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        file_paths = filedialog.askopenfilenames(
            title="Select LIF file(s)",
            filetypes=[("LIF files", "*.lif"), ("All files", "*.*")]
        )
        
        root.destroy()
        return file_paths
    
    def process_lif_file(self, lif_file_path):
        """Process a single LIF file"""
        try:
            print(f"Processing {os.path.basename(lif_file_path)}")
            lif_file = LifFile(lif_file_path)

            base_name = os.path.splitext(os.path.basename(lif_file_path))[0]
            base_name = self.clean_filename(base_name)
            output_dir = f'./{base_name}_extracted_pngs'
            os.makedirs(output_dir, exist_ok=True)

            all_images = list(lif_file.get_iter_image())
            print(f"Found {len(all_images)} images in LIF file")

            for img_index, lif_image in enumerate(all_images):
                series_name = self.clean_filename(lif_image.name)
                print(f"Processing image {img_index + 1}/{len(all_images)}: {series_name}")
                dims = lif_image.dims
                print(f"  Dimensions: {dims}")
                print(f"  Channels: {lif_image.channels}")
                
                # Create series folder first, then channel subfolders within it
                series_dir = os.path.join(output_dir, series_name)
                os.makedirs(series_dir, exist_ok=True)

                # Ask the user (once per distinct channel setup) which physical
                # channel is nuclei/mbp/pillar, instead of assuming c=0/1/2.
                channel_role = self.resolve_channel_mapping(lif_file, img_index, series_name, lif_image)
                print(f"  Confirmed channel mapping: {channel_role}")

                # Create channel subfolders only for roles actually in use
                for channel, role in channel_role.items():
                    if role == "skip":
                        continue
                    channel_dir = os.path.join(output_dir, series_name, role)
                    os.makedirs(channel_dir, exist_ok=True)

                for channel, role in channel_role.items():
                    if role == "skip":
                        print(f"  Channel {channel} skipped by user")
                        continue
                    if channel >= lif_image.channels:
                        print(f"  Channel {channel} not available (only {lif_image.channels} channels)")
                        continue

                    # The "trim first/last z-slice" behaviour belongs to the mbp
                    # ROLE, not to a hard-coded channel index - so it follows
                    # wherever the user says mbp actually is.
                    if role == "mbp":
                        z_total = dims.z if hasattr(dims, 'z') else 1
                        z_used = z_total - 2 if z_total > 2 else z_total
                        print(f"  Processing channel {channel} (mbp) - using {z_used} of {z_total} z-slices")
                        z_start = 1  # Skip first slice (index 0)
                        z_end = (dims.z if hasattr(dims, 'z') else 1) - 1  # Skip last slice
                        z_range = range(z_start, z_end) if z_end > z_start else range(0, 1)
                    else:
                        # For other roles, use all z-slices
                        z_range = range(dims.z if hasattr(dims, 'z') else 1)

                    for z in z_range:
                        for t in range(dims.t if hasattr(dims, 't') else 1):
                            try:
                                pil_image = lif_image.get_frame(t=t, z=z, c=channel)
                                np_image = np.array(pil_image)

                                if np_image.max() > np_image.min():
                                    normalized_image = (np_image - np_image.min()) / (np_image.max() - np_image.min())
                                else:
                                    normalized_image = np_image

                                # Colour convention (yellow for pillar, cyan otherwise)
                                # also now follows the confirmed role.
                                if role == "pillar":
                                    colored_image = self.yellow_cmap(normalized_image)
                                    channel_name = "yellow"
                                else:  # nuclei / mbp
                                    colored_image = self.cyan_cmap(normalized_image)
                                    channel_name = "cyan"

                                rgb_image = (colored_image[:, :, :3] * 255).astype(np.uint8)
                                img = Image.fromarray(rgb_image, 'RGB')

                                # Determine channel directory with series subfolder and filename
                                channel_dir = os.path.join(output_dir, series_name, role)
                                filename = f'{series_name}_c{channel}_{channel_name}_z{z}_t{t}.png'
                                filename = self.clean_filename(filename)
                                filepath = os.path.join(channel_dir, filename)

                                img.save(filepath, format='PNG')
                                print(f"    Saved: {os.path.join(role, series_name, filename)}")

                            except Exception as e:
                                print(f"    Error processing frame: {e}")
            
            print(f"Finished processing {os.path.basename(lif_file_path)}")
            return True
            
        except Exception as e:
            print(f"Error processing {lif_file_path}: {e}")
            return False

    def main(self):
        """Main function - handles file selection and processing"""
        print("LIF to PNG Extractor")
        print("=" * 50)

        # Browse for LIF files
        lif_files = self.browse_lif_file()
        
        if not lif_files:
            print("No files selected!")
            input("Press Enter to exit...")
            return
        
        print(f"Selected {len(lif_files)} LIF file(s):")
        for lif_file in lif_files:
            print(f"  - {os.path.basename(lif_file)}")
        
        print("\nStarting processing...")
        for lif_file in lif_files:
            if lif_file.lower().endswith('.lif'):
                self.process_lif_file(lif_file)
            else:
                print(f"Skipping non-LIF file: {os.path.basename(lif_file)}")
        
        print("\nProcessing complete!")
        if self._tk_root is not None:
            self._tk_root.destroy()
        input("Press Enter to exit...")

if __name__ == "__main__":
    processor = LIFProcessor()
    processor.main()