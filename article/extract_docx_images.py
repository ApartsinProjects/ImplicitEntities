import zipfile
import os

docx_path = r"E:\Projects\ImplicitEntities\Report\Final Project Book - draft.docx"
output_dir = r"E:\Projects\ImplicitEntities\article\figures\docx_images"

os.makedirs(output_dir, exist_ok=True)

with zipfile.ZipFile(docx_path, 'r') as z:
    media_files = [f for f in z.namelist() if f.startswith('word/media/')]
    print(f"Found {len(media_files)} files in word/media/:")
    for f in media_files:
        print(f"  {f}")
        filename = os.path.basename(f)
        data = z.read(f)
        out_path = os.path.join(output_dir, filename)
        with open(out_path, 'wb') as out:
            out.write(data)
        print(f"    -> extracted to {out_path} ({len(data)} bytes)")

print(f"\nDone. Extracted {len(media_files)} files to {output_dir}")
