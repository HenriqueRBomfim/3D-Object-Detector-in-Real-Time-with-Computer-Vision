"use client";
import Image from "next/image";
import { useState } from "react";

export default function Home() {
	const [file, setFile] = useState<File | null>(null);
	const [preview, setPreview] = useState<string>("");
	const [result, setResult] = useState<any>(null);
	const [loading, setLoading] = useState(false);

	function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
		const selected = e.target.files?.[0] ?? null;
		if (!selected) return;
		setFile(selected);
		setPreview(URL.createObjectURL(selected));
		setResult(null);
	}
	const options = [
		"Exemplo 1",
		"Option 2",
		"Option 3",
		"Option 4",
		"Option 5",
		"Option 6",
		"Option 7",
		"Option 8",
		"Option 9",
		"Option 10",
	];
	// 2) mapeamento de cada opção para a sua imagem
	const optionImages: Record<string, string> = {
		"Exemplo 1": "/dropdown/option1.png",
		"Option 2": "/dropdown/option2.png",
		"Option 3": "/dropdown/option3.png",
		"Option 4": "/dropdown/option4.png",
		"Option 5": "/dropdown/option5.png",
		"Option 6": "/dropdown/option6.png",
		"Option 7": "/dropdown/option7.png",
		"Option 8": "/dropdown/option8.png",
		"Option 9": "/dropdown/option9.png",
		"Option 10": "/dropdown/option10.png",
	};
	const [selectedOption, setSelectedOption] = useState(options[0]);
	function handleOptionChange(e: React.ChangeEvent<HTMLSelectElement>) {
		setSelectedOption(e.target.value);
	}
	async function handleSubmit() {
		if (!file) return;
		setLoading(true);
		const formData = new FormData();
		formData.append("image", file);
		// try {
		// 	const res = await fetch("/api/pose", {
		// 		method: "POST",
		// 		body: formData,
		// 	});
		// 	const data = await res.json();
		// 	setResult(data);
		// } catch (err) {
		// 	console.error(err);
		// } finally {
		setLoading(false);
		// }
	}
	return (
		<div className="min-h-screen p-8 sm:p-20 flex flex-col items-center gap-6">
			<h1 className="text-3xl font-bold">Pose Corrector</h1>
			<div className="mt-4 flex justify-center items-center gap-4 w-full">
				<input
					type="file"
					accept="image/*"
					onChange={handleFileChange}
					className="border rounded px-2 py-1"
				/>
				<select
					value={selectedOption}
					onChange={handleOptionChange}
					className="self-center  border rounded px-2 py-1 self"
				>
					{options.map((opt) => (
						<option key={opt} value={opt}>
							{opt}
						</option>
					))}
				</select>
			</div>

			{preview && (
				<div className="mt-4 flex gap-4 items-start">
					<Image
						aria-hidden
						src={preview}
						alt="Preview"
						width={400}
						height={800}
						style={{ width: "auto", height: "auto" }}
						className="border rounded max-w-xs"
					/>
					{/* imagem associada à opção selecionada */}
					<Image
						aria-hidden
						src={optionImages[selectedOption]}
						alt={selectedOption}
						width={400}
						height={800}
						style={{ width: "auto", height: "auto" }}
						className="border rounded max-w-xs"
					/>
				</div>
			)}

			<button
				onClick={handleSubmit}
				disabled={!file || loading}
				className="mt-4 px-4 py-2 bg-blue-600 text-white rounded disabled:opacity-50"
			>
				{loading ? "Enviando..." : "Enviar Imagem"}
			</button>

			{result && (
				<div className="mt-6 w-full max-w-lg">
					<h2 className="text-2xl font-semibold mb-2">
						Resultado do Back-end
					</h2>
					<pre className="bg-gray-100 dark:bg-gray-800 p-4 rounded overflow-auto">
						{JSON.stringify(result, null, 2)}
					</pre>
				</div>
			)}
		</div>
	);
}
