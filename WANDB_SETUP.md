# WandB Setup Guide

This guide explains how to configure WandB (Weights & Biases) for logging your experiments.

## Option 1: Using `wandb login` (Recommended)

This is the easiest and most secure method:

### Step 1: Install WandB

```bash
pip install wandb
```

### Step 2: Login with API Key

```bash
wandb login
```

When prompted, paste your API key. The key will be saved securely in `~/.netrc`.

**To get your API key:**
1. Go to https://wandb.ai/settings
2. Scroll to "API keys"
3. Copy your API key

### Step 3: Verify Login

```bash
wandb status
```

You should see your username and API key status.

## Option 2: Environment Variable

You can set the API key as an environment variable:

### Temporary (for current session)

```bash
export WANDB_API_KEY="your-api-key-here"
./run_experiments.sh lr_schedule
```

### Permanent (add to your shell profile)

Add to `~/.bashrc` or `~/.zshrc`:

```bash
export WANDB_API_KEY="your-api-key-here"
```

Then reload:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

## Option 3: Set in Script (Not Recommended)

You can uncomment and set in `run_experiments.sh`:

```bash
WANDB_API_KEY="your-api-key-here"
```

⚠️ **Warning**: This is less secure as the key will be visible in the script file.

## Option 4: Using WandB Config File

Create `~/.config/wandb/settings`:

```ini
[default]
api_key = your-api-key-here
```

## WandB Entity and Project

The script is configured with:
- **Project**: `orlm_sft_experiments` (set in `run_experiments.sh`)
- **Entity**: Empty (uses your default WandB account)

To change the entity, edit `run_experiments.sh`:

```bash
WANDB_ENTITY="your-entity-name"  # Your WandB username or team name
```

## Verify Setup

After setting up, you can verify by running a quick test:

```bash
python -c "import wandb; wandb.init(project='test'); wandb.log({'test': 1}); wandb.finish()"
```

Check https://wandb.ai to see if the test run appears.

## Troubleshooting

### Error: "wandb: ERROR Not logged in"

**Solution**: Run `wandb login` and paste your API key.

### Error: "wandb: ERROR API key not found"

**Solution**: 
1. Check if `~/.netrc` exists and contains your API key
2. Or set `WANDB_API_KEY` environment variable
3. Or run `wandb login` again

### Error: "wandb: ERROR Invalid API key"

**Solution**: 
1. Get a fresh API key from https://wandb.ai/settings
2. Run `wandb login` again with the new key

### WandB Not Logging During Training

**Check**:
1. WandB is installed: `pip list | grep wandb`
2. You're logged in: `wandb status`
3. The script has `--report_to "wandb"` (already configured)

## Security Best Practices

1. ✅ **Recommended**: Use `wandb login` (stores key securely)
2. ✅ **Acceptable**: Use environment variable
3. ⚠️ **Avoid**: Hardcoding API key in scripts
4. ⚠️ **Never**: Commit API keys to git repositories

## Viewing Results

Once experiments start running:

1. Go to https://wandb.ai
2. Find project: `orlm_sft_experiments`
3. View metrics, charts, and logs for each experiment run

Each experiment will appear as a separate run with metrics like:
- Training loss
- Evaluation metrics for each dataset
- Learning rate schedule
- Training progress

