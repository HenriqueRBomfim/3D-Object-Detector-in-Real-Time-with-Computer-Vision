"use client";
import { setEngine } from "crypto";
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
		"Pose 1",
		"Pose 2",
		"Pose 3",
		"Pose 4",
		"Pose 5",
		"Pose 6",
		"Pose 7",
		"Pose 8",
		"Pose 9",
		"Pose 10",
		"Pose 11",
		"Pose 12",
		"Pose 13",
		"Pose 14",
		"Pose 15",
		"Pose 16",
		"Pose 17",
		"Pose 18",
		"Pose 19",
		"Pose 20",
		"Pose 21",
	];
	// 2) mapeamento de cada opção para a sua imagem
	const optionImages: Record<string, string> = {
		"Pose 1": "/pose1.png",
		"Pose 2": "/pose2.png",
		"Pose 3": "/pose3.png",
		"Pose 4": "/pose4.png",
		"Pose 5": "/pose5.png",
		"Pose 6": "/pose6.png",
		"Pose 7": "/pose7.png",
		"Pose 8": "/pose8.png",
		"Pose 9": "/pose9.png",
		"Pose 10": "/pose10.png",
		"Pose 11": "/pose11.png",
		"Pose 12": "/pose12.png",
		"Pose 13": "/pose13.png",
		"Pose 14": "/pose14.png",
		"Pose 15": "/pose15.png",
		"Pose 16": "/pose16.png",
		"Pose 17": "/pose17.png",
		"Pose 18": "/pose18.png",
		"Pose 19": "/pose19.png",
		"Pose 20": "/pose20.png",
		"Pose 21": "/pose18.png",
	};
	const [selectedOption, setSelectedOption] = useState(options[0]);
	const [percentageFinal, setPercentageFinal] = useState(0);
	function handleOptionChange(e: React.ChangeEvent<HTMLSelectElement>) {
		setSelectedOption(e.target.value);
	}
	interface DataItem {
		selectedOption: string;
		[key: string]: any;
	}
	async function handleSubmit() {
		if (!file) return;
		setLoading(true);

		const formData = new FormData();
		formData.append("image", file);
		formData.append("target_pose", selectedOption); // << ADICIONADO

		try {
			const res = await fetch("http://0.0.0.0:8000/api/pose", {
				method: "POST",
				body: formData,
			});
			const data = await res.json();
			console.log(data);
			console.log(data.predictions, "data 0");
			const perc = Math.round(data.predictions[selectedOption] * 100);
			setPercentageFinal(perc);
			setResult(data);
			console.log(perc);
		} catch (err) {
			console.error(err);
			alert("Erro ao processar a imagem.");
		} finally {
			setLoading(false);
		}
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
					<p className="text-2xl font-semibold">
						Você fez a pose{" "}
						<span className="underline">{selectedOption}</span>{" "}
						<span className="text-blue-600">
							{percentageFinal}%
						</span>{" "}
						certo.
					</p>
				</div>
			)}
		</div>
	);
}
