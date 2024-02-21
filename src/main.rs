use ai_nft_io::{NftAction, NftError, NftEvent};
use diffusers::pipelines::stable_diffusion;
use diffusers::transformers::clip;
use gclient::GearApi;
use gear_core::ids::ProgramId as GearCoreProgramId;
use gsdk::{
    gp::{Decode, Encode},
    metadata::runtime_types::{
        gear_core::{ids::ProgramId as GearRuntimeProgramId, message::user::UserMessage},
        pallet_gear::pallet::Event as GearEvent,
        vara_runtime::RuntimeEvent,
    },
    Api,
};
use ipfs_api::{IpfsApi, IpfsClient, TryFromUri};
use std::{collections::HashSet, error::Error};
use std::{fs, io::Cursor};
use tch::{nn::Module, Device, Kind, Tensor};

const PROGRAM_ID_RAW: [u8; 32] =
    hex_literal::hex!("444194bf695bb9654695ea30eacd47d0803b1ca06317f11e5d257a5b6d4af6d6");
const PROGRAM_ID: GearRuntimeProgramId = GearRuntimeProgramId(PROGRAM_ID_RAW);

const GUIDANCE_SCALE: f64 = 7.5;

async fn generate_pictute(prompt: String) -> Result<String, Box<dyn Error>> {
    let clip_weights = "data/pytorch_model.safetensors".to_owned();
    let vae_weights = "data/vae.safetensors".to_owned();
    let unet_weights = "data/unet.safetensors".to_owned();
    let vocab_file = "data/bpe_simple_vocab_16e6.txt".to_owned();
    let final_image = "sd_final.jpg".to_owned();

    let cpu = vec![];
    let n_steps = 30;
    let seed = 32;

    tch::maybe_init_cuda();

    println!("Cuda available: {}", tch::Cuda::is_available());
    println!("Cudnn available: {}", tch::Cuda::cudnn_is_available());
    println!("MPS available: {}", tch::utils::has_mps());

    // sliced_attention_size, height, width
    let sd_config = stable_diffusion::StableDiffusionConfig::v1_5(None, None, None);

    let device_setup = diffusers::utils::DeviceSetup::new(cpu);
    let clip_device = device_setup.get("clip");
    let vae_device = device_setup.get("vae");
    let unet_device = device_setup.get("unet");
    let scheduler = sd_config.build_scheduler(n_steps);

    let tokenizer = clip::Tokenizer::create(vocab_file, &sd_config.clip)?;
    println!("Running with prompt \"{prompt}\".");
    let tokens = tokenizer.encode(&prompt)?;
    let tokens: Vec<i64> = tokens.into_iter().map(|x| x as i64).collect();
    let tokens = Tensor::from_slice(&tokens).view((1, -1)).to(clip_device);
    let uncond_tokens = tokenizer.encode("")?;
    let uncond_tokens: Vec<i64> = uncond_tokens.into_iter().map(|x| x as i64).collect();
    let uncond_tokens = Tensor::from_slice(&uncond_tokens)
        .view((1, -1))
        .to(clip_device);

    let no_grad_guard = tch::no_grad_guard();

    println!("Building the Clip transformer.");
    let text_model = sd_config.build_clip_transformer(&clip_weights, clip_device)?;
    let text_embeddings = text_model.forward(&tokens);
    let uncond_embeddings = text_model.forward(&uncond_tokens);
    let text_embeddings = Tensor::cat(&[uncond_embeddings, text_embeddings], 0).to(unet_device);

    println!("Building the autoencoder.");
    let vae = sd_config.build_vae(&vae_weights, vae_device)?;
    println!("Building the unet.");
    let unet = sd_config.build_unet(&unet_weights, unet_device, 4)?;

    let bsize = 1;
    tch::manual_seed(seed);
    let mut latents = Tensor::randn(
        [bsize, 4, sd_config.height / 8, sd_config.width / 8],
        (Kind::Float, unet_device),
    );

    // scale the initial noise by the standard deviation required by the scheduler
    latents *= scheduler.init_noise_sigma();

    for (timestep_index, &timestep) in scheduler.timesteps().iter().enumerate() {
        println!("Timestep {timestep_index}/{n_steps}");
        let latent_model_input = Tensor::cat(&[&latents, &latents], 0);

        let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep);
        let noise_pred = unet.forward(&latent_model_input, timestep as f64, &text_embeddings);
        let noise_pred = noise_pred.chunk(2, 0);
        let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);
        let noise_pred = noise_pred_uncond + (noise_pred_text - noise_pred_uncond) * GUIDANCE_SCALE;
        latents = scheduler.step(&noise_pred, timestep, &latents);
    }

    println!("Generating the final image.");
    let latents = latents.to(vae_device);
    let image = vae.decode(&(&latents / 0.18215));
    let image = (image / 2 + 0.5).clamp(0., 1.).to_device(Device::Cpu);
    let image = (image * 255.).to_kind(Kind::Uint8);
    tch::vision::image::save(&image, final_image.clone())?;

    drop(no_grad_guard);

    let client = IpfsClient::from_str("https://ipfs.gear-tech.io/api/v0")?;
    let data = Cursor::new(fs::read(&final_image)?);

    println!("start upload");
    let response = client.add(data).await?;
    println!("end upload");
    Ok(format!(
        "https://ipfs-gw.gear-tech.io/ipfs/{hash}",
        hash = response.hash
    ))
}

#[allow(unused_must_use)]
async fn process_event(
    visited_message_ids: &mut HashSet<[u8; 32]>,
    event: RuntimeEvent,
) -> Result<(), Box<dyn Error>> {
    if let RuntimeEvent::Gear(GearEvent::UserMessageSent {
        message:
            UserMessage {
                id,
                source,
                payload,
                ..
            },
        ..
    }) = &event
    {
        if *source == PROGRAM_ID {
            let mut buffer = &payload.0[..];
            let result = <Result<NftEvent, NftError>>::decode(&mut buffer);
            if let Ok(Ok(NftEvent::PhaseOneOfMintDone {
                minter_to_personal_id: (minter, personal_id),
                words,
            })) = result
            {
                if visited_message_ids.contains(&id.0) {
                    println!("skip duplicated");
                    return Ok(());
                }

                visited_message_ids.insert(id.0);

                println!("{event:?}");
                let prompt = words.join(" ");
                println!("{prompt:?}");

                let api = GearApi::vara_testnet().await?;

                //tokio::spawn(async move {
                if let Ok(img_link) = generate_pictute(prompt).await {
                    dbg!(&img_link);
                    let payload = NftAction::SecondPhaseOfMint {
                        minter,
                        personal_id,
                        img_link,
                    }
                    .encode();

                    let program_id = GearCoreProgramId::from(PROGRAM_ID_RAW);
                    let gas_info = api
                        .calculate_handle_gas(None, program_id, payload.clone(), 0, true)
                        .await;
                    if let Ok(gas_info) = gas_info {
                        let ret = api
                            .send_message_bytes(program_id, payload, gas_info.min_limit, 0)
                            .await;
                        dbg!(ret);
                    }
                }
                //});
            }
        }
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    //let api = Api::new(Some("ws://127.0.0.1:9944")).await?;
    let api = Api::new(Some("wss://testnet.vara.network:443")).await?;
    let mut events = api.events().await?;
    let mut visited_message_ids = HashSet::new();
    loop {
        if let Some(Ok(events)) = events.next().await {
            for event in events {
                process_event(&mut visited_message_ids, event).await?;
            }
        }
    }
}
